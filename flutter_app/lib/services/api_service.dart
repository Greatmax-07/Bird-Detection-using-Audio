import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/audio_input.dart';
import '../models/detection_result.dart';
import '../models/health_status.dart';
import 'settings_service.dart';

class ApiException implements Exception {
  const ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class ApiService {
  ApiService({String? baseUrl}) : _baseUrlOverride = baseUrl;

  static const Duration _healthTimeout = Duration(seconds: 8);
  static const Duration _predictTimeout = Duration(minutes: 5);

  final String? _baseUrlOverride;

  String get baseUrl => _baseUrlOverride ?? SettingsService.instance.baseUrl;

  Uri _uri(String path) {
    final root = baseUrl.trim().replaceFirst(RegExp(r'/+$'), '');
    return Uri.parse('$root$path');
  }

  Future<HealthStatus> health() async {
    try {
      final response = await http.get(_uri('/health')).timeout(_healthTimeout);
      final body = _decodeObject(response.body);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException(
          body?['error']?.toString() ??
              'Server returned ${response.statusCode}.',
        );
      }
      if (body == null) {
        throw const ApiException(
          'The server returned an invalid health response.',
        );
      }
      return HealthStatus.fromJson(body);
    } on TimeoutException {
      throw const ApiException('The server did not respond in time.');
    } on http.ClientException catch (error) {
      throw ApiException('Could not reach the server: ${error.message}');
    }
  }

  Future<DetectionResult> predict(AudioInput input) async {
    if (!input.hasData) {
      throw const ApiException('No audio data was provided.');
    }

    final request = http.MultipartRequest('POST', _uri('/predict'));

    try {
      if (input.bytes != null) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            input.bytes!,
            filename: input.filename,
          ),
        );
      } else if (kIsWeb || input.isBlobUrl) {
        final data = await http
            .get(Uri.parse(input.path!))
            .timeout(const Duration(seconds: 30));
        if (data.statusCode < 200 || data.statusCode >= 300) {
          throw const ApiException('Could not read the recorded audio.');
        }
        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            data.bodyBytes,
            filename: input.filename,
          ),
        );
      } else {
        request.files.add(
          await http.MultipartFile.fromPath(
            'file',
            input.path!,
            filename: input.filename,
          ),
        );
      }

      final streamed = await request.send().timeout(_predictTimeout);
      final response = await http.Response.fromStream(streamed);
      final body = _decodeObject(response.body);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw ApiException(
          body?['error']?.toString() ??
              'The server could not analyze this audio (HTTP ${response.statusCode}).',
        );
      }
      if (body == null) {
        throw const ApiException('The server returned an invalid response.');
      }

      return DetectionResult.fromJson(
        body,
        sourceName: input.displayName,
        analyzedAt: DateTime.now(),
      );
    } on TimeoutException {
      throw const ApiException(
        'Analysis timed out. Try a shorter clip or check the server.',
      );
    } on http.ClientException catch (error) {
      throw ApiException('Could not reach the server: ${error.message}');
    }
  }

  Map<String, dynamic>? _decodeObject(String body) {
    if (body.trim().isEmpty) return null;
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map) return Map<String, dynamic>.from(decoded);
    } catch (_) {
      return null;
    }
    return null;
  }
}
