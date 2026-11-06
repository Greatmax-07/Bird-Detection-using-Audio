import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/detection_result.dart';

class HistoryService extends ChangeNotifier {
  HistoryService._();

  static final HistoryService instance = HistoryService._();

  static const String _storageKey = 'birdear.history';
  static const int _maxEntries = 100;

  final List<DetectionResult> _entries = [];
  bool _loaded = false;

  List<DetectionResult> get entries => List.unmodifiable(_entries);
  bool get isEmpty => _entries.isEmpty;
  bool get loaded => _loaded;

  Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    _entries.clear();
    if (raw != null && raw.isNotEmpty) {
      try {
        final decoded = jsonDecode(raw);
        if (decoded is List) {
          for (final item in decoded.whereType<Map>()) {
            try {
              _entries.add(
                DetectionResult.fromJson(Map<String, dynamic>.from(item)),
              );
            } catch (_) {
              // Skip a malformed entry instead of losing the whole history.
            }
          }
        }
      } catch (_) {
        // Corrupt storage is treated as an empty history.
      }
    }
    _loaded = true;
    notifyListeners();
  }

  Future<void> add(DetectionResult result) async {
    _entries.insert(0, result);
    if (_entries.length > _maxEntries) {
      _entries.removeRange(_maxEntries, _entries.length);
    }
    await _persist();
    notifyListeners();
  }

  Future<void> removeAt(int index) async {
    if (index < 0 || index >= _entries.length) return;
    _entries.removeAt(index);
    await _persist();
    notifyListeners();
  }

  Future<void> clear() async {
    _entries.clear();
    await _persist();
    notifyListeners();
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    final payload = jsonEncode(_entries.map((e) => e.toJson()).toList());
    await prefs.setString(_storageKey, payload);
  }
}
