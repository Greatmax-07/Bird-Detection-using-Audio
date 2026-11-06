import 'package:intl/intl.dart';

String formatClock(double seconds) {
  final total = seconds.isFinite ? seconds.round() : 0;
  final minutes = total ~/ 60;
  final secs = total % 60;
  return '$minutes:${secs.toString().padLeft(2, '0')}';
}

String formatPlaybackTime(double seconds) {
  final total = seconds.isFinite && seconds > 0 ? seconds.round() : 0;
  final minutes = total ~/ 60;
  final secs = total % 60;
  return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
}

String formatDuration(double seconds) {
  if (!seconds.isFinite || seconds <= 0) return '0s';
  final total = seconds.round();
  if (total < 60) return '${total}s';
  final minutes = total ~/ 60;
  final secs = total % 60;
  if (secs == 0) return '${minutes}m';
  return '${minutes}m ${secs}s';
}

String formatPercent(double value) {
  final clamped = value.isFinite ? value.clamp(0.0, 1.0) : 0.0;
  return '${(clamped * 100).round()}%';
}

String formatTimestamp(DateTime value) {
  return DateFormat('d MMM yyyy, h:mm a').format(value.toLocal());
}

String formatShortDate(DateTime value) {
  final local = value.toLocal();
  final now = DateTime.now();
  final today = DateUtils.dateOnly(now);
  final day = DateUtils.dateOnly(local);
  final difference = today.difference(day).inDays;
  if (difference == 0) {
    return 'Today, ${DateFormat('h:mm a').format(local)}';
  }
  if (difference == 1) {
    return 'Yesterday, ${DateFormat('h:mm a').format(local)}';
  }
  return DateFormat('d MMM, h:mm a').format(local);
}

class DateUtils {
  static DateTime dateOnly(DateTime value) {
    return DateTime(value.year, value.month, value.day);
  }
}
