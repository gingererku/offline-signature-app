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

  final ImagePicker _picker = ImagePicker();
  bool _isBusy = false;
  double? _accuracyPct;
  double? _rawSimValue;
  bool? _isSamePerson;

  // عتبة القبول بعد إلغاء الانحياز الموجب (افتراضياً 0.75)
  double _threshold = 0.75;

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

  // إظهار قائمة لاختيار الكاميرا أو المعرض (الجاليري)
  Future<ImageSource?> _chooseImageSource() async {
    return showModalBottomSheet<ImageSource>(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) => SafeArea(
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Colors.indigo),
              title: const Text("التقاط صورة بالكاميرا", style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () => Navigator.pop(ctx, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library, color: Colors.teal),
              title: const Text("اختيار صورة من الجاليري", style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () => Navigator.pop(ctx, ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
  }

  Future<Uint8List> _preprocess(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception("Invalid image");

    decoded = img.bakeOrientation(decoded);

    // قص 10% من الحواف لتجاهل أطراف الورقة والظل الجانبي
    int marginX = (decoded.width * 0.10).toInt();
    int marginY = (decoded.height * 0.10).toInt();
    decoded = img.copyCrop(
      decoded,
      x: marginX,
      y: marginY,
      width: decoded.width - 2 * marginX,
      height: decoded.height - 2 * marginY,
    );

    img.Image gray = img.grayscale(decoded);

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
          gray.setPixelRgb(x, y, 255, 255, 255);
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        } else {
          gray.setPixelRgb(x, y, 0, 0, 0);
        }
      }
    }

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
    return _centerAndNormalize(out!);
  }

  // طرح المتوسط قبل التطبيع لإلغاء الانحياز الموجب وإظهار الفروق الحقيقية بين الإمضاءات
  List<double> _centerAndNormalize(List<double> v) {
    double mean = v.reduce((a, b) => a + b) / v.length;
    List<double> centered = v.map((x) => x - mean).toList();
    double norm = sqrt(centered.fold(0.0, (s, x) => s + x * x));
    return norm == 0 ? centered : centered.map((x) => x / norm).toList();
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
      _showAlert("تنبيه", "يرجى إدخال الإمضاء رقم ${index} أولاً بالترتيب.");
      return;
    }

    final source = await _chooseImageSource();
    if (source == null) return;

    final picked = await _picker.pickImage(source: source, imageQuality: 85);
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
        meanPrev = _centerAndNormalize(meanPrev);

        double sim = _cosineSim(meanPrev, newEmb);
        if (sim < _threshold) {
          setState(() => _isBusy = false);
          _showAlert(
            "⚠️ إمضاء غير متطابق!",
            "درجة التشابه: ${sim.toStringAsFixed(3)} (أقل من الحد المطلوب ${_threshold.toStringAsFixed(2)})\nيرجى اختيار إمضاءة مطابقة.",
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

    final source = await _chooseImageSource();
    if (source == null) return;

    final picked = await _picker.pickImage(source: source, imageQuality: 85);
    if (picked == null) return;

    setState(() => _isBusy = true);
    try {
      Uint8List procBytes = await _preprocess(File(picked.path));
      setState(() {
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
      meanRef = _centerAndNormalize(meanRef);

      double rawSim = _cosineSim(meanRef, testEmb);
      double score = (rawSim * 100).clamp(0.0, 100.0);

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
                        Text(_threshold.toStringAsFixed(2), style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.indigo)),
                      ],
                    ),
                    Slider(
                      value: _threshold,
                      min: 0.40,
                      max: 0.99,
                      divisions: 59,
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
            const Text("1️⃣ تسجيل 3 إمضاءات متطابقة (كاميرا أو جاليري):", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
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
                                  const Icon(Icons.add_a_photo, color: Colors.indigo, size: 28),
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
                          Text("اضغط لاختيار أو تصوير الإمضاء الرابع", style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
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