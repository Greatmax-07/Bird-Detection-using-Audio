import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsService extends ChangeNotifier {
  SettingsService._();

  static final SettingsService instance = SettingsService._();

  static const String _baseUrlKey = 'birdear.base_url';

  String _baseUrl = defaultBaseUrl();
  bool _loaded = false;

  String get baseUrl => _baseUrl;
  bool get loaded => _loaded;

  static String defaultBaseUrl() {
    if (kIsWeb) return 'http://127.0.0.1:8000';
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getString(_baseUrlKey);
    if (stored != null && stored.trim().isNotEmpty) {
      _baseUrl = _normalize(stored);
    }
    _loaded = true;
    notifyListeners();
  }

  Future<void> updateBaseUrl(String value) async {
    final normalized = _normalize(value);
    if (normalized == _baseUrl) return;
    _baseUrl = normalized;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlKey, normalized);
    notifyListeners();
  }

  Future<void> resetBaseUrl() async {
    await updateBaseUrl(defaultBaseUrl());
  }

  static String _normalize(String value) {
    var result = value.trim();
    while (result.endsWith('/')) {
      result = result.substring(0, result.length - 1);
    }
    return result;
  }
}
