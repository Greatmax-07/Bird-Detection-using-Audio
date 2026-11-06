import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/detection_result.dart';
import '../utils/formatters.dart';

class EventTimeline extends StatelessWidget {
  const EventTimeline({
    super.key,
    required this.events,
    required this.durationSeconds,
  });

  final List<DetectionEvent> events;
  final double durationSeconds;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final duration = durationSeconds > 0
        ? durationSeconds
        : events.isEmpty
        ? 1.0
        : events.last.endTime;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final width = constraints.maxWidth;
            return SizedBox(
              height: 12,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Container(
                      decoration: BoxDecoration(
                        color: scheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(999),
                      ),
                    ),
                  ),
                  for (final event in events)
                    Positioned(
                      left:
                          (event.startTime / duration).clamp(0.0, 1.0) * width,
                      width: math.max(
                        4,
                        (event.endTime - event.startTime) / duration * width,
                      ),
                      top: 0,
                      bottom: 0,
                      child: Container(
                        decoration: BoxDecoration(
                          color: scheme.primary.withValues(
                            alpha:
                                0.45 + 0.5 * event.confidence.clamp(0.0, 1.0),
                          ),
                          borderRadius: BorderRadius.circular(999),
                        ),
                      ),
                    ),
                ],
              ),
            );
          },
        ),
        const SizedBox(height: 12),
        for (final event in events.take(5))
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                Icon(Icons.graphic_eq_rounded, size: 16, color: scheme.primary),
                const SizedBox(width: 8),
                Text(
                  '${formatClock(event.startTime)} - ${formatClock(event.endTime)}',
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                Text(
                  formatPercent(event.confidence),
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: scheme.onSurfaceVariant,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        if (events.length > 5)
          Text(
            '+${events.length - 5} more event${events.length - 5 == 1 ? '' : 's'}',
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
      ],
    );
  }
}
