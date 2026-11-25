import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:image/image.dart' as img;

void main() => runApp(const MaterialApp(
      home: OfflineSignatureApp(),
      debugShowCheckedModeBanner: false,
    ));

class OfflineSignatureApp extends StatefulWidget {
  const OfflineSignatureApp({super.key});
  @override
  State<OfflineSignatureApp> createState() => _OfflineSignatureAppState();
}

class _OfflineSignatureAppState extends State<OfflineSignatureApp> {
  ClassificationModel? _model;
  String? _loadError;
  final List<Uint8List?> _refProcessed = [null, null, null];
  final List<List<double>?> _refEmbeddings = [null, null, null];
  Uint8List? _testProcessed;
  File? _testRawFile;

  final ImagePicker _picker = ImagePicker();
  bool _isBusy = false;
  double? _accuracyPct;
  double? _rawSimValue;
  bool? _isSamePerson;

  // عتبة قابلة للتعديل مباشرة من التطبيق (افتراضياً 0.96)
  double _threshold = 0.96;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _model = await PytorchLite.loadClassificationModel(
        "assets/signature_mobile.ptl",
        224,
        224,
        128,
        labelPath: "assets/labels.txt",
      );
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _loadError = e.toString());
    }
  }

  Future<Uint8List> _preprocess(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception("Invalid image");

    decoded = img.bakeOrientation(decoded);

    // 1. قص 12% من الحواف لتجاهل أطراف الورقة والظل الجانبي
    int marginX = (decoded.width * 0.12).toInt();
    int marginY = (decoded.height * 0.12).toInt();
    decoded = img.copyCrop(
      decoded,
      x: marginX,
      y: marginY,
      width: decoded.width - 2 * marginX,
      height: decoded.height - 2 * marginY,
    );

    img.Image gray = img.grayscale(decoded);

    // 2. حساب متوسط الإضاءة وعزل الحبر الغامق فقط (تجاهل الظلال الخفيفة)
    int sum = 0;
    for (var p in gray) {
      sum += p.r.toInt();
    }
    int meanVal = (sum / (gray.width * gray.height)).toInt();
    int thresh = (meanVal * 0.68).toInt();

    int minX = gray.width, minY = gray.height, maxX = 0, maxY = 0;
    bool found = false;

    for (int y = 0; y < gray.height; y++) {
      for (int x = 0; x < gray.width; x++) {
        int val = gray.getPixel(x, y).r.toInt();
        if (val < thresh) {
          gray.setPixelRgb(x, y, 255, 255, 255); // خط الإمضاء أبيض
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        } else {
          gray.setPixelRgb(x, y, 0, 0, 0); // الخلفية سوداء
        }
      }
    }

    // 3. قص الإمضاء بدقة مع هامش أمان بسيط (10 بكسل)
    if (found && (maxX - minX) > 10 && (maxY - minY) > 10) {
      int pad = 10;
      int cx = max(0, minX - pad);
      int cy = max(0, minY - pad);
      int cw = min(gray.width - cx, (maxX - minX) + 2 * pad);
      int ch = min(gray.height - cy, (maxY - minY) + 2 * pad);
      gray = img.copyCrop(gray, x: cx, y: cy, width: cw, height: ch);
    }

    img.Image resized = img.copyResize(gray, width: 224, height: 224);
    return Uint8List.fromList(img.encodePng(resized));
  }

  Future<List<double>> _getEmbedding(Uint8List processedBytes) async {
    List<double>? out = await _model!.getImagePredictionList(
      processedBytes,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    );
    return _l2Normalize(out!);
  }

  List<double> _l2Normalize(List<double> v) {
    double norm = sqrt(v.fold(0.0, (s, x) => s + x * x));
    return norm == 0 ? v : v.map((x) => x / norm).toList();
  }

  double _cosineSim(List<double> a, List<double> b) {
    double dot = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  Future<void> _captureReference(int index) async {
    if (_model == null || _isBusy) return;
    if (index > 0 && _refEmbeddings[index - 1] == null) {
      _showAlert("تنبيه", "يرجى تصوير الإمضاء رقم ${index} أولاً بالترتيب.");
      return;
    }

    final picked = await _picker.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (picked == null) return;

    setState(() => _isBusy = true);
    try {
      Uint8List procBytes = await _preprocess(File(picked.path));
      List<double> newEmb = await _getEmbedding(procBytes);

      if (index > 0) {
        List<List<double>> prevEmbs = [];
        for (int i = 0; i < index; i++) {
          prevEmbs.add(_refEmbeddings[i]!);
        }
        List<double> meanPrev = List.generate(
          newEmb.length,
          (j) => prevEmbs.map((e) => e[j]).reduce((a, b) => a + b) / prevEmbs.length,
        );
        meanPrev = _l2Normalize(meanPrev);

        double sim = _cosineSim(meanPrev, newEmb);
        if (sim < _threshold) {
          setState(() => _isBusy = false);
          _showAlert(
            "⚠️ إمضاء غير متطابق!",
            "درجة التشابه: ${sim.toStringAsFixed(4)} (أقل من الحد المطلوب ${_threshold.toStringAsFixed(3)})\nيرجى توقيع إمضاءة مطابقة.",
          );
          return;
        }
      }

      setState(() {
        _refProcessed[index] = procBytes;
        _refEmbeddings[index] = newEmb;
        _accuracyPct = null;
        _rawSimValue = null;
        _isBusy = false;
      });
    } catch (e) {
      setState(() => _isBusy = false);
      _showAlert("خطأ أثناء المعالجة", e.toString());
    }
  }

  Future<void> _captureTest() async {
    if (_refEmbeddings.contains(null)) {
      _showAlert("تنبيه", "أكمل تسجيل الإمضاءات المرجعية الثلاثة أولاً.");
      return;
    }
    final picked = await _picker.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (picked == null) return;

    setState(() => _isBusy = true);
    try {
      File raw = File(picked.path);
      Uint8List procBytes = await _preprocess(raw);
      setState(() {
        _testRawFile = raw;
        _testProcessed = procBytes;
        _accuracyPct = null;
        _rawSimValue = null;
        _isBusy = false;
      });
    } catch (e) {
      setState(() => _isBusy = false);
      _showAlert("خطأ", e.toString());
    }
  }

  Future<void> _verifyFinal() async {
    if (_testProcessed == null || _refEmbeddings.contains(null)) return;
    setState(() => _isBusy = true);

    try {
      List<double> testEmb = await _getEmbedding(_testProcessed!);
      List<double> meanRef = List.generate(
        testEmb.length,
        (j) => (_refEmbeddings[0]![j] + _refEmbeddings[1]![j] + _refEmbeddings[2]![j]) / 3.0,
      );
      meanRef = _l2Normalize(meanRef);

      double rawSim = _cosineSim(meanRef, testEmb);
      // معايرة النسبة المئوية بحيث 0.90 = 0% و 1.00 = 100%
      double score = (((rawSim - 0.90) / 0.10) * 100).clamp(0.0, 100.0);

      setState(() {
        _rawSimValue = rawSim;
        _accuracyPct = score;
        _isSamePerson = rawSim >= _threshold;
        _isBusy = false;
      });
    } catch (e) {
      setState(() => _isBusy = false);
      _showAlert("خطأ أثناء التحقق", e.toString());
    }
  }

  void _resetAll() {
    setState(() {
      for (int i = 0; i < 3; i++) {
        _refProcessed[i] = null;
        _refEmbeddings[i] = null;
      }
      _testProcessed = null;
      _testRawFile = null;
      _accuracyPct = null;
      _rawSimValue = null;
    });
  }

  void _showAlert(String title, String msg) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(title, style: const TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
        content: Text(msg, style: const TextStyle(fontSize: 16)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("حسناً")),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("التحقق البنكي (Offline AI)"),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
        actions: [
          IconButton(onPressed: _resetAll, icon: const Icon(Icons.refresh), tooltip: "بدء من جديد"),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_loadError != null)
              Container(
                padding: const EdgeInsets.all(10),
                color: Colors.red.shade100,
                child: Text("خطأ: $_loadError", style: const TextStyle(color: Colors.red)),
              ),
            // شريط ضبط حساسية الموديل لحظياً
            Card(
              color: Colors.indigo.shade50,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text("حساسية القبول (Threshold):", style: TextStyle(fontWeight: FontWeight.bold)),
                        Text(_threshold.toStringAsFixed(3), style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.indigo)),
                      ],
                    ),
                    Slider(
                      value: _threshold,
                      min: 0.90,
                      max: 0.999,
                      divisions: 99,
                      onChanged: (v) {
                        setState(() {
                          _threshold = v;
                          if (_rawSimValue != null) {
                            _isSamePerson = _rawSimValue! >= _threshold;
                          }
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Text("1️⃣ تسجيل 3 إمضاءات متطابقة (معاينة AI):", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Row(
              children: List.generate(3, (i) {
                return Expanded(
                  child: GestureDetector(
                    onTap: () => _captureReference(i),
                    child: Container(
                      height: 110,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      decoration: BoxDecoration(
                        border: Border.all(color: _refProcessed[i] != null ? Colors.green : Colors.indigo, width: 2),
                        borderRadius: BorderRadius.circular(10),
                        color: Colors.black87,
                      ),
                      child: _refProcessed[i] != null
                          ? ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.memory(_refProcessed[i]!, fit: BoxFit.contain))
                          : Container(
                              color: Colors.grey.shade100,
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(Icons.camera_alt, color: Colors.indigo, size: 28),
                                  const SizedBox(height: 4),
                                  Text("إمضاء ${i + 1}", style: const TextStyle(fontWeight: FontWeight.bold)),
                                ],
                              ),
                            ),
                    ),
                  ),
                );
              }),
            ),
            const Divider(height: 24),
            const Text("2️⃣ إمضاء الاختبار (للتحقق):", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            GestureDetector(
              onTap: _captureTest,
              child: Container(
                height: 140,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.deepOrange, width: 2),
                  borderRadius: BorderRadius.circular(12),
                  color: _testProcessed != null ? Colors.black87 : Colors.grey.shade100,
                ),
                child: _testProcessed != null
                    ? ClipRRect(borderRadius: BorderRadius.circular(10), child: Image.memory(_testProcessed!, fit: BoxFit.contain))
                    : const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.document_scanner, color: Colors.deepOrange, size: 40),
                          SizedBox(height: 6),
                          Text("اضغط لتصوير الإمضاء الرابع", style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 14),
            if (_isBusy) const Center(child: CircularProgressIndicator()),
            if (_accuracyPct != null && _rawSimValue != null)
              Container(
                padding: const EdgeInsets.all(14),
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  color: _isSamePerson! ? Colors.green.shade100 : Colors.red.shade100,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: _isSamePerson! ? Colors.green : Colors.red, width: 2),
                ),
                child: Column(
                  children: [
                    Text(
                      _isSamePerson! ? "✅ هو نفس الشخص (توقيع مطابق)" : "❌ شخص مختلف / توقيع مزور",
                      style: const TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      "نسبة التطابق: ${_accuracyPct!.toStringAsFixed(1)}%  (Raw: ${_rawSimValue!.toStringAsFixed(4)})",
                      style: const TextStyle(fontSize: 15),
                    ),
                  ],
                ),
              ),
            SizedBox(
              height: 50,
              child: ElevatedButton(
                onPressed: (_testProcessed == null || _isBusy) ? null : _verifyFinal,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.indigo, foregroundColor: Colors.white),
                child: const Text("🔍 تحقق من التوقيع الآن", style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}