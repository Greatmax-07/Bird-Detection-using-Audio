import 'package:birdear/models/audio_input.dart';
import 'package:birdear/models/detection_result.dart';
import 'package:birdear/utils/formatters.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('detection result round-trips through JSON', () {
    final result = DetectionResult.fromJson(
      {
        'predictions': [
          {'species': 'House Crow', 'confidence': 0.91},
        ],
        'species_detected': [
          {
            'species': 'House Crow',
            'event_count': 2,
            'estimated_count': 2,
            'confidence': 0.91,
            'count_note': 'Separate stretches of calling.',
            'events': [
              {
                'start_time': 3.0,
                'end_time': 9.0,
                'confidence': 0.91,
                'segments': 2,
                'windows': 2,
              },
              {
                'start_time': 15.0,
                'end_time': 18.0,
                'confidence': 0.72,
                'segments': 1,
                'windows': 1,
              },
            ],
          },
        ],
        'no_confident_detection': false,
        'total_approximate_count': 2,
        'count_note': 'Separate stretches of calling.',
        'duration_seconds': 21.0,
      },
      sourceName: 'garden.wav',
      analyzedAt: DateTime.parse('2026-01-02T03:04:05Z'),
    );

    expect(result.hasDetections, isTrue);
    expect(result.eventCount, 2);
    expect(result.speciesDetected.first.events.length, 2);
    expect(result.speciesDetected.first.events.first.segments, 2);

    final restored = DetectionResult.fromJson(result.toJson());
    expect(restored.sourceName, 'garden.wav');
    expect(restored.speciesDetected.first.species, 'House Crow');
    expect(restored.speciesDetected.first.events.last.startTime, 15.0);
    expect(restored.totalApproximateCount, 2);
  });

  test('no confident detection response parses safely', () {
    final result = DetectionResult.fromJson({
      'predictions': [
        {'species': 'Shikra', 'confidence': 0.18},
      ],
      'species_detected': [],
      'no_confident_detection': true,
      'total_approximate_count': 0,
      'count_note': 'No birds exceeded the threshold.',
      'duration_seconds': 12.0,
    });

    expect(result.hasDetections, isFalse);
    expect(result.noConfidentDetection, isTrue);
    expect(result.predictions.single.species, 'Shikra');
    expect(result.eventCount, 0);
  });

  test('formatters handle edge values', () {
    expect(formatClock(0), '0:00');
    expect(formatClock(65), '1:05');
    expect(formatPlaybackTime(0), '00:00');
    expect(formatPlaybackTime(65), '01:05');
    expect(formatPlaybackTime(3605), '60:05');
    expect(formatPercent(0.876), '88%');
    expect(formatDuration(90), '1m 30s');
  });

  test('audio input exposes playback mime types', () {
    expect(const AudioInput(filename: 'clip.WAV').mimeType, 'audio/wav');
    expect(const AudioInput(filename: 'clip.mp3').mimeType, 'audio/mpeg');
    expect(const AudioInput(filename: 'clip.m4a').mimeType, 'audio/mp4');
    expect(const AudioInput(filename: 'clip.bin').mimeType, isNull);
  });
}
