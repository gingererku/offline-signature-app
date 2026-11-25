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
  final List<File?> _refImages = [null, null, null];
  final List<List<double>?> _refEmbeddings = [null, null, null];
  File? _testImage;

  final ImagePicker _picker = ImagePicker();
  bool _isBusy = false;
  double? _accuracyPct;
  bool? _isSamePerson;

  final double _threshold = 0.89;

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
      if (mounted) {
        setState(() {
          _loadError = e.toString();
        });
      }
    }
  }

  Future<Uint8List> _preprocess(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception("Invalid image");

    decoded = img.bakeOrientation(decoded);
    img.Image gray = img.grayscale(decoded);

    int sum = 0;
    for (var p in gray) {
      sum += p.r.toInt();
    }
    int thresh = ((sum / (gray.width * gray.height)) * 0.78).toInt();

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

    if (found && maxX > minX && maxY > minY) {
      gray = img.copyCrop(gray, x: minX, y: minY, width: maxX - minX, height: maxY - minY);
    }

    img.Image resized = img.copyResize(gray, width: 224, height: 224);
    return Uint8List.fromList(img.encodePng(resized));
  }

  Future<List<double>> _getEmbedding(File file) async {
    Uint8List bytes = await _preprocess(file);
    List<double>? out = await _model!.getImagePredictionList(
      bytes,
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
      File file = File(picked.path);
      List<double> newEmb = await _getEmbedding(file);

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
            "هذه الإمضاءة لا تشبه الإمضاءات السابقة!\nيرجى إعادة المحاولة وتوقيع إمضاءة مطابقة.",
          );
          return;
        }
      }

      setState(() {
        _refImages[index] = file;
        _refEmbeddings[index] = newEmb;
        _accuracyPct = null;
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

    setState(() {
      _testImage = File(picked.path);
      _accuracyPct = null;
    });
  }

  Future<void> _verifyFinal() async {
    if (_testImage == null || _refEmbeddings.contains(null)) return;
    setState(() => _isBusy = true);

    try {
      List<double> testEmb = await _getEmbedding(_testImage!);
      List<double> meanRef = List.generate(
        testEmb.length,
        (j) => (_refEmbeddings[0]![j] + _refEmbeddings[1]![j] + _refEmbeddings[2]![j]) / 3.0,
      );
      meanRef = _l2Normalize(meanRef);

      double rawSim = _cosineSim(meanRef, testEmb);
      double score = (((rawSim - 0.70) / 0.30) * 100).clamp(0.0, 100.0);

      setState(() {
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
        _refImages[i] = null;
        _refEmbeddings[i] = null;
      }
      _testImage = null;
      _accuracyPct = null;
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
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_loadError != null)
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 12),
                color: Colors.red.shade100,
                child: Text("خطأ في تحميل الموديل:\n$_loadError", style: const TextStyle(color: Colors.red)),
              ),
            if (_model == null && _loadError == null)
              const Padding(
                padding: EdgeInsets.only(bottom: 12),
                child: Row(
                  children: [
                    SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                    SizedBox(width: 10),
                    Text("جاري تهيئة نموذج الذكاء الاصطناعي..."),
                  ],
                ),
              ),
            const Text("1️⃣ تسجيل 3 إمضاءات متطابقة (المرجع):", style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Row(
              children: List.generate(3, (i) {
                return Expanded(
                  child: GestureDetector(
                    onTap: () => _captureReference(i),
                    child: Container(
                      height: 115,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      decoration: BoxDecoration(
                        border: Border.all(color: _refImages[i] != null ? Colors.green : Colors.indigo, width: 2),
                        borderRadius: BorderRadius.circular(10),
                        color: Colors.grey.shade100,
                      ),
                      child: _refImages[i] != null
                          ? ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.file(_refImages[i]!, fit: BoxFit.cover))
                          : Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const Icon(Icons.camera_alt, color: Colors.indigo, size: 30),
                                const SizedBox(height: 4),
                                Text("إمضاء ${i + 1}", style: const TextStyle(fontWeight: FontWeight.bold)),
                              ],
                            ),
                    ),
                  ),
                );
              }),
            ),
            const Divider(height: 30),
            const Text("2️⃣ إمضاء الاختبار (للتحقق):", style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            GestureDetector(
              onTap: _captureTest,
              child: Container(
                height: 150,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.deepOrange, width: 2),
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.grey.shade100,
                ),
                child: _testImage != null
                    ? ClipRRect(borderRadius: BorderRadius.circular(10), child: Image.file(_testImage!, fit: BoxFit.contain))
                    : const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.document_scanner, color: Colors.deepOrange, size: 42),
                          SizedBox(height: 6),
                          Text("اضغط لتصوير الإمضاء الرابع", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 18),
            if (_isBusy) const Center(child: CircularProgressIndicator()),
            if (_accuracyPct != null)
              Container(
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 14),
                decoration: BoxDecoration(
                  color: _isSamePerson! ? Colors.green.shade100 : Colors.red.shade100,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: _isSamePerson! ? Colors.green : Colors.red, width: 2),
                ),
                child: Column(
                  children: [
                    Text(
                      _isSamePerson! ? "✅ هو نفس الشخص (توقيع مطابق)" : "❌ شخص مختلف / توقيع مزور",
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 6),
                    Text("نسبة التطابق: ${_accuracyPct!.toStringAsFixed(1)}%", style: const TextStyle(fontSize: 16)),
                  ],
                ),
              ),
            SizedBox(
              height: 52,
              child: ElevatedButton(
                onPressed: (_testImage == null || _isBusy) ? null : _verifyFinal,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.indigo, foregroundColor: Colors.white),
                child: const Text("🔍 تحقق من التوقيع الآن", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}