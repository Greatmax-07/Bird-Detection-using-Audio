import 'package:flutter/material.dart';

import '../models/detection_result.dart';
import '../utils/formatters.dart';
import 'event_timeline.dart';

class SpeciesCard extends StatelessWidget {
  const SpeciesCard({
    super.key,
    required this.detection,
    required this.rank,
    required this.durationSeconds,
  });

  final SpeciesDetection detection;
  final int rank;
  final double durationSeconds;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final accent = confidenceColor(detection.confidence);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: scheme.primaryContainer,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    '$rank',
                    style: TextStyle(
                      color: scheme.onPrimaryContainer,
                      fontWeight: FontWeight.w800,
                      fontSize: 16,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        detection.species,
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                          height: 1.2,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '${detection.eventCount} ${detection.eventCount == 1 ? 'event' : 'events'}'
                        '  ·  ${detection.estimatedCount} estimated occurrence${detection.estimatedCount == 1 ? '' : 's'}',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: accent.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    formatPercent(detection.confidence),
                    style: TextStyle(
                      color: accent,
                      fontWeight: FontWeight.w800,
                      fontSize: 13,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: detection.confidence.clamp(0.0, 1.0),
                minHeight: 8,
                backgroundColor: scheme.surfaceContainerHighest,
                valueColor: AlwaysStoppedAnimation<Color>(accent),
              ),
            ),
            if (detection.events.isNotEmpty) ...[
              const SizedBox(height: 18),
              Divider(
                color: scheme.outlineVariant.withValues(alpha: 0.5),
                height: 1,
              ),
              const SizedBox(height: 16),
              EventTimeline(
                events: detection.events,
                durationSeconds: durationSeconds,
              ),
            ],
            if (detection.countNote.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                detection.countNote,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                  height: 1.4,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

Color confidenceColor(double value) {
  if (value >= 0.75) return const Color(0xFF2E7D32);
  if (value >= 0.5) return const Color(0xFFF9A825);
  return const Color(0xFFEF6C00);
}
