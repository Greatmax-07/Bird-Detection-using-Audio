import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

class RecordedAudio {
  const RecordedAudio({
    required this.path,
    required this.filename,
    required this.duration,
  });

  final String path;
  final String filename;
  final Duration duration;
}

class RecordingService {
  RecordingService();

  final AudioRecorder _recorder = AudioRecorder();

  bool get isRecording => _isRecording;
  bool _isRecording = false;

  Future<bool> ensurePermission() async {
    return _recorder.hasPermission();
  }

  Future<void> start() async {
    if (_isRecording) return;

    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final filename = 'birdear_$timestamp.wav';

    String path;
    if (kIsWeb) {
      path = filename;
    } else {
      final directory = await getTemporaryDirectory();
      path = '${directory.path}/$filename';
    }

    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        bitRate: 128000,
        sampleRate: 48000,
        numChannels: 1,
        autoGain: true,
      ),
      path: path,
    );
    _isRecording = true;
  }

  Future<RecordedAudio?> stop(Duration duration) async {
    if (!_isRecording) return null;
    final path = await _recorder.stop();
    _isRecording = false;
    if (path == null || path.isEmpty) return null;
    return RecordedAudio(
      path: path,
      filename: 'birdear_recording.wav',
      duration: duration,
    );
  }

  Future<void> cancel() async {
    if (!_isRecording) return;
    await _recorder.cancel();
    _isRecording = false;
  }

  Future<void> dispose() async {
    await _recorder.dispose();
  }
}
