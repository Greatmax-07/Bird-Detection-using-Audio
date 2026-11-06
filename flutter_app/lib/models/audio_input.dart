import 'dart:typed_data';

class AudioInput {
  const AudioInput({required this.filename, this.path, this.bytes});

  final String filename;
  final String? path;
  final Uint8List? bytes;

  bool get hasData => bytes != null || (path != null && path!.isNotEmpty);

  bool get isBlobUrl => path != null && path!.startsWith('blob:');

  String get extension {
    final dot = filename.lastIndexOf('.');
    if (dot < 0 || dot == filename.length - 1) return '';
    return filename.substring(dot + 1).toLowerCase();
  }

  String get displayName => filename.isEmpty ? 'Audio clip' : filename;

  String? get mimeType {
    switch (extension) {
      case 'wav':
        return 'audio/wav';
      case 'mp3':
        return 'audio/mpeg';
      case 'flac':
        return 'audio/flac';
      case 'ogg':
        return 'audio/ogg';
      case 'm4a':
        return 'audio/mp4';
      case 'aac':
        return 'audio/aac';
      case 'webm':
        return 'audio/webm';
      default:
        return null;
    }
  }
}
