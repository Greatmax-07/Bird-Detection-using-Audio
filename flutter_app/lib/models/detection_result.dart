class DetectionEvent {
  const DetectionEvent({
    required this.startTime,
    required this.endTime,
    required this.confidence,
    required this.segments,
  });

  final double startTime;
  final double endTime;
  final double confidence;
  final int segments;

  factory DetectionEvent.fromJson(Map<String, dynamic> json) {
    return DetectionEvent(
      startTime: _asDouble(json['start_time']),
      endTime: _asDouble(json['end_time']),
      confidence: _asDouble(json['confidence']),
      segments: _asInt(json['segments'] ?? json['windows'], 1),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'start_time': startTime,
      'end_time': endTime,
      'confidence': confidence,
      'segments': segments,
      'windows': segments,
    };
  }
}

class Prediction {
  const Prediction({required this.species, required this.confidence});

  final String species;
  final double confidence;

  factory Prediction.fromJson(Map<String, dynamic> json) {
    return Prediction(
      species: _asString(json['species']),
      confidence: _asDouble(json['confidence']),
    );
  }

  Map<String, dynamic> toJson() {
    return {'species': species, 'confidence': confidence};
  }
}

class SpeciesDetection {
  const SpeciesDetection({
    required this.species,
    required this.eventCount,
    required this.estimatedCount,
    required this.countNote,
    required this.confidence,
    required this.events,
  });

  final String species;
  final int eventCount;
  final int estimatedCount;
  final String countNote;
  final double confidence;
  final List<DetectionEvent> events;

  factory SpeciesDetection.fromJson(Map<String, dynamic> json) {
    final rawEvents = json['events'];
    final events = rawEvents is List
        ? rawEvents
              .whereType<Map>()
              .map((e) => DetectionEvent.fromJson(Map<String, dynamic>.from(e)))
              .toList()
        : <DetectionEvent>[];
    return SpeciesDetection(
      species: _asString(json['species']),
      eventCount: _asInt(json['event_count'], events.length),
      estimatedCount: _asInt(json['estimated_count'], events.length),
      countNote: _asString(json['count_note']),
      confidence: _asDouble(json['confidence']),
      events: events,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'species': species,
      'event_count': eventCount,
      'estimated_count': estimatedCount,
      'count_note': countNote,
      'confidence': confidence,
      'events': events.map((e) => e.toJson()).toList(),
    };
  }
}

class DetectionResult {
  const DetectionResult({
    required this.predictions,
    required this.speciesDetected,
    required this.noConfidentDetection,
    required this.totalApproximateCount,
    required this.countNote,
    required this.durationSeconds,
    required this.sourceName,
    required this.analyzedAt,
  });

  final List<Prediction> predictions;
  final List<SpeciesDetection> speciesDetected;
  final bool noConfidentDetection;
  final int totalApproximateCount;
  final String countNote;
  final double durationSeconds;
  final String sourceName;
  final DateTime analyzedAt;

  bool get hasDetections => speciesDetected.isNotEmpty;

  int get eventCount => speciesDetected.fold(0, (sum, s) => sum + s.eventCount);

  factory DetectionResult.fromJson(
    Map<String, dynamic> json, {
    String sourceName = 'Recording',
    DateTime? analyzedAt,
  }) {
    final rawPredictions = json['predictions'];
    final rawSpecies = json['species_detected'];

    return DetectionResult(
      predictions: rawPredictions is List
          ? rawPredictions
                .whereType<Map>()
                .map((e) => Prediction.fromJson(Map<String, dynamic>.from(e)))
                .toList()
          : <Prediction>[],
      speciesDetected: rawSpecies is List
          ? rawSpecies
                .whereType<Map>()
                .map(
                  (e) =>
                      SpeciesDetection.fromJson(Map<String, dynamic>.from(e)),
                )
                .toList()
          : <SpeciesDetection>[],
      noConfidentDetection: json['no_confident_detection'] == true,
      totalApproximateCount: _asInt(json['total_approximate_count']),
      countNote: _asString(json['count_note']),
      durationSeconds: _asDouble(json['duration_seconds']),
      sourceName: _asString(json['source_name'], sourceName),
      analyzedAt: json['analyzed_at'] is String
          ? DateTime.tryParse(json['analyzed_at'] as String) ??
                analyzedAt ??
                DateTime.now()
          : analyzedAt ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'predictions': predictions.map((e) => e.toJson()).toList(),
      'species_detected': speciesDetected.map((e) => e.toJson()).toList(),
      'no_confident_detection': noConfidentDetection,
      'total_approximate_count': totalApproximateCount,
      'count_note': countNote,
      'duration_seconds': durationSeconds,
      'source_name': sourceName,
      'analyzed_at': analyzedAt.toIso8601String(),
    };
  }

  DetectionResult copyWith({String? sourceName, DateTime? analyzedAt}) {
    return DetectionResult(
      predictions: predictions,
      speciesDetected: speciesDetected,
      noConfidentDetection: noConfidentDetection,
      totalApproximateCount: totalApproximateCount,
      countNote: countNote,
      durationSeconds: durationSeconds,
      sourceName: sourceName ?? this.sourceName,
      analyzedAt: analyzedAt ?? this.analyzedAt,
    );
  }
}

double _asDouble(dynamic value, [double fallback = 0]) {
  if (value is num) return value.toDouble();
  return double.tryParse('$value') ?? fallback;
}

int _asInt(dynamic value, [int fallback = 0]) {
  if (value is num) return value.toInt();
  return int.tryParse('$value') ?? fallback;
}

String _asString(dynamic value, [String fallback = '']) {
  if (value is String) return value;
  if (value == null) return fallback;
  return value.toString();
}
