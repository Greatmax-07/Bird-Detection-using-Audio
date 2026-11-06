import 'dart:async';
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/audio_input.dart';
import '../models/detection_result.dart';
import '../utils/formatters.dart';
import 'info_banner.dart';

class AudioPlayerCard extends StatefulWidget {
  const AudioPlayerCard({super.key, required this.input, required this.result});

  final AudioInput input;
  final DetectionResult result;

  @override
  State<AudioPlayerCard> createState() => _AudioPlayerCardState();
}

class _AudioPlayerCardState extends State<AudioPlayerCard> {
  late final AudioPlayer _player;

  StreamSubscription<PlayerState>? _stateSubscription;
  StreamSubscription<Duration>? _positionSubscription;
  StreamSubscription<Duration>? _durationSubscription;
  StreamSubscription<void>? _completeSubscription;

  Duration _position = Duration.zero;
  Duration? _nativeDuration;
  bool _loading = true;
  bool _prepared = false;
  bool _playing = false;
  bool _dragging = false;
  double _dragSeconds = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _player = AudioPlayer();
    _stateSubscription = _player.onPlayerStateChanged.listen((state) {
      if (!mounted) return;
      setState(() => _playing = state == PlayerState.playing);
    }, onError: _handleStreamError);
    _positionSubscription = _player.onPositionChanged.listen((position) {
      if (!mounted || _dragging) return;
      setState(() => _position = position);
    }, onError: _handleStreamError);
    _durationSubscription = _player.onDurationChanged.listen((duration) {
      if (!mounted || duration <= Duration.zero) return;
      setState(() => _nativeDuration = duration);
    }, onError: _handleStreamError);
    _completeSubscription = _player.onPlayerComplete.listen(
      (_) => _handleComplete(),
      onError: _handleStreamError,
    );
    _prepare();
  }

  @override
  void dispose() {
    unawaited(_stateSubscription?.cancel());
    unawaited(_positionSubscription?.cancel());
    unawaited(_durationSubscription?.cancel());
    unawaited(_completeSubscription?.cancel());
    unawaited(_player.dispose());
    super.dispose();
  }

  double get _fallbackSeconds {
    final value = widget.result.durationSeconds;
    return value.isFinite && value > 0 ? value : 0;
  }

  double get _durationSeconds {
    final native = _nativeDuration;
    if (native != null && native.inMilliseconds > 0) {
      return native.inMilliseconds / 1000;
    }
    return _fallbackSeconds;
  }

  double get _positionSeconds => _position.inMilliseconds / 1000;

  double get _sliderMax => math.max(_durationSeconds, 0.01);

  double get _sliderValue {
    final value = _dragging ? _dragSeconds : _positionSeconds;
    return value.clamp(0.0, _sliderMax).toDouble();
  }

  List<_PlaybackEvent> get _events {
    final events = <_PlaybackEvent>[];
    for (final species in widget.result.speciesDetected) {
      for (final event in species.events) {
        events.add(_PlaybackEvent(species.species, event));
      }
    }
    events.sort((a, b) => a.event.startTime.compareTo(b.event.startTime));
    return events;
  }

  Future<void> _prepare() async {
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
      });
    }

    try {
      await _player.setReleaseMode(ReleaseMode.stop);
      await _player.setSource(_sourceFor(widget.input));
      final duration = await _player.getDuration();
      if (!mounted) return;
      setState(() {
        _loading = false;
        _prepared = true;
        if (duration != null && duration > Duration.zero) {
          _nativeDuration = duration;
        }
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _prepared = false;
        _error = _friendlyError(error);
      });
    }
  }

  Source _sourceFor(AudioInput input) {
    final bytes = input.bytes;
    if (bytes != null && bytes.isNotEmpty) {
      return BytesSource(bytes, mimeType: input.mimeType);
    }

    final path = input.path;
    if (path != null && path.isNotEmpty) {
      if (kIsWeb || input.isBlobUrl) {
        return UrlSource(path, mimeType: input.mimeType);
      }
      return DeviceFileSource(path, mimeType: input.mimeType);
    }

    throw StateError('No audio data is available for playback.');
  }

  Future<void> _toggle() async {
    if (_loading) return;

    if (!_prepared) {
      await _prepare();
      return;
    }

    try {
      if (_playing) {
        await _player.pause();
        return;
      }

      final atEnd =
          _player.state == PlayerState.completed ||
          (_durationSeconds > 0 && _positionSeconds >= _durationSeconds - 0.05);
      if (atEnd) {
        await _player.seek(Duration.zero);
        if (mounted) setState(() => _position = Duration.zero);
      }

      await _player.resume();
      if (mounted) setState(() => _playing = true);
    } catch (error) {
      _showError(_friendlyError(error));
    }
  }

  Future<void> _handleComplete() async {
    if (!mounted) return;
    try {
      await _player.stop();
    } catch (_) {
      // The player may already have released its resources.
    }
    if (!mounted) return;
    setState(() {
      _playing = false;
      _position = Duration.zero;
    });
  }

  Future<void> _seekTo(double seconds) async {
    if (!_prepared) return;
    final targetSeconds = seconds.clamp(0.0, _durationSeconds).toDouble();
    final target = Duration(milliseconds: (targetSeconds * 1000).round());

    try {
      await _player.seek(target);
      if (!mounted) return;
      setState(() {
        _dragging = false;
        _position = target;
      });
    } catch (error) {
      if (mounted) setState(() => _dragging = false);
      _showError(_friendlyError(error));
    }
  }

  void _handleStreamError(Object error, [StackTrace? stackTrace]) {
    if (!mounted) return;
    setState(() {
      _loading = false;
      _playing = false;
      _error = _friendlyError(error);
    });
  }

  String _friendlyError(Object error) {
    final message = error.toString().replaceFirst('Exception: ', '');
    final compact = message.length > 220
        ? '${message.substring(0, 217)}...'
        : message;
    return 'Could not play this audio. $compact';
  }

  void _showError(String message) {
    if (!mounted) return;
    setState(() => _error = message);
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final events = _events;
    final activeEvents = events
        .where(
          (item) =>
              _sliderValue >= item.event.startTime &&
              _sliderValue < item.event.endTime,
        )
        .toList();
    final durationLabel = _durationSeconds > 0
        ? formatPlaybackTime(_durationSeconds)
        : '--:--';

    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: scheme.primaryContainer.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.graphic_eq_rounded,
                    size: 20,
                    color: scheme.onPrimaryContainer,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Listen to your recording',
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        widget.result.sourceName,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                if (events.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 9,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: scheme.primaryContainer.withValues(alpha: 0.55),
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      '${events.length} call${events.length == 1 ? '' : 's'}',
                      style: TextStyle(
                        color: scheme.onPrimaryContainer,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                IconButton.filled(
                  onPressed: _loading ? null : _toggle,
                  tooltip: _playing ? 'Pause' : 'Play',
                  iconSize: 26,
                  style: IconButton.styleFrom(
                    minimumSize: const Size(52, 52),
                    backgroundColor: scheme.primary,
                    foregroundColor: scheme.onPrimary,
                    disabledBackgroundColor: scheme.primaryContainer,
                    disabledForegroundColor: scheme.onPrimaryContainer,
                  ),
                  icon: _loading
                      ? SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: scheme.onPrimaryContainer,
                          ),
                        )
                      : Icon(
                          _playing
                              ? Icons.pause_rounded
                              : Icons.play_arrow_rounded,
                        ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    children: [
                      SliderTheme(
                        data: SliderTheme.of(context).copyWith(
                          trackHeight: 4,
                          thumbShape: const RoundSliderThumbShape(
                            enabledThumbRadius: 7,
                          ),
                          overlayShape: const RoundSliderOverlayShape(
                            overlayRadius: 16,
                          ),
                          activeTrackColor: scheme.primary,
                          inactiveTrackColor: scheme.surfaceContainerHighest,
                        ),
                        child: Slider(
                          value: _sliderValue,
                          max: _sliderMax,
                          onChangeStart: _prepared && _durationSeconds > 0
                              ? (value) {
                                  setState(() {
                                    _dragging = true;
                                    _dragSeconds = value;
                                  });
                                }
                              : null,
                          onChanged: _prepared && _durationSeconds > 0
                              ? (value) {
                                  setState(() {
                                    _dragging = true;
                                    _dragSeconds = value;
                                  });
                                }
                              : null,
                          onChangeEnd: _prepared && _durationSeconds > 0
                              ? _seekTo
                              : null,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: Row(
                          children: [
                            Text(
                              formatPlaybackTime(_sliderValue),
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: scheme.onSurfaceVariant,
                                fontFeatures: const [
                                  FontFeature.tabularFigures(),
                                ],
                              ),
                            ),
                            const Spacer(),
                            Text(
                              durationLabel,
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: scheme.onSurfaceVariant,
                                fontFeatures: const [
                                  FontFeature.tabularFigures(),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              InfoBanner(
                icon: Icons.error_outline_rounded,
                color: scheme.error,
                message: _error!,
              ),
            ],
            const SizedBox(height: 12),
            _PlaybackTimeline(
              events: events,
              durationSeconds: _durationSeconds,
              positionSeconds: _sliderValue,
            ),
            const SizedBox(height: 8),
            Text(
              activeEvents.isNotEmpty
                  ? 'Active: ${activeEvents.map((item) => '${item.species} ${formatPlaybackTime(item.event.startTime)}-${formatPlaybackTime(item.event.endTime)}').join(' · ')}'
                  : events.isEmpty
                  ? 'No call intervals were detected in this clip.'
                  : 'No detected call at this position.',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.bodySmall?.copyWith(
                color: activeEvents.isNotEmpty
                    ? scheme.primary
                    : scheme.onSurfaceVariant,
                fontWeight: activeEvents.isNotEmpty
                    ? FontWeight.w700
                    : FontWeight.w500,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PlaybackEvent {
  const _PlaybackEvent(this.species, this.event);

  final String species;
  final DetectionEvent event;
}

class _PlaybackTimeline extends StatelessWidget {
  const _PlaybackTimeline({
    required this.events,
    required this.durationSeconds,
    required this.positionSeconds,
  });

  final List<_PlaybackEvent> events;
  final double durationSeconds;
  final double positionSeconds;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final total = durationSeconds > 0
        ? durationSeconds
        : events.isEmpty
        ? 1.0
        : events.last.event.endTime;

    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        return SizedBox(
          height: 14,
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
              for (final item in events)
                Positioned(
                  left: _x(item.event.startTime, total, width),
                  width: math.max(
                    3.0,
                    _x(item.event.endTime - item.event.startTime, total, width),
                  ),
                  top: 0,
                  bottom: 0,
                  child: Container(
                    decoration: BoxDecoration(
                      color:
                          positionSeconds >= item.event.startTime &&
                              positionSeconds < item.event.endTime
                          ? scheme.primary
                          : scheme.primary.withValues(alpha: 0.38),
                      border:
                          positionSeconds >= item.event.startTime &&
                              positionSeconds < item.event.endTime
                          ? Border.all(color: scheme.onPrimary, width: 1.5)
                          : null,
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                ),
              if (durationSeconds > 0)
                Positioned(
                  left: _x(positionSeconds, total, width) - 1,
                  top: -2,
                  bottom: -2,
                  child: Container(
                    width: 2,
                    decoration: BoxDecoration(
                      color: scheme.onSurface,
                      borderRadius: BorderRadius.circular(999),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  double _x(double seconds, double total, double width) {
    final safeTotal = total > 0 ? total : 1.0;
    return (seconds / safeTotal).clamp(0.0, 1.0).toDouble() * width;
  }
}
