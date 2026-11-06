import 'dart:async';

import 'package:flutter/material.dart';

import '../models/audio_input.dart';
import '../services/recording_service.dart';
import '../services/settings_service.dart';
import '../utils/formatters.dart';
import '../widgets/info_banner.dart';
import 'results_page.dart';

class RecordPage extends StatefulWidget {
  const RecordPage({super.key});

  @override
  State<RecordPage> createState() => _RecordPageState();
}

class _RecordPageState extends State<RecordPage> {
  final RecordingService _recorder = RecordingService();

  Timer? _timer;
  DateTime? _startedAt;
  Duration _elapsed = Duration.zero;

  bool _recording = false;
  bool _starting = false;
  bool _stopping = false;
  String? _error;

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_starting || _stopping) return;
    if (_recording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    setState(() {
      _starting = true;
      _error = null;
    });
    try {
      final granted = await _recorder.ensurePermission();
      if (!granted) {
        if (!mounted) return;
        setState(() {
          _starting = false;
          _error = 'Microphone permission was denied. Enable it in system settings and try again.';
        });
        return;
      }

      await _recorder.start();
      if (!mounted) return;
      setState(() {
        _starting = false;
        _recording = true;
        _elapsed = Duration.zero;
      });
      _startedAt = DateTime.now();
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (!mounted || _startedAt == null) return;
        setState(() => _elapsed = DateTime.now().difference(_startedAt!));
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _starting = false;
        _recording = false;
        _error = 'Could not start recording: $error';
      });
    }
  }

  Future<void> _stopRecording() async {
    _timer?.cancel();
    setState(() => _stopping = true);
    final elapsed = _elapsed;
    try {
      final recorded = await _recorder.stop(elapsed);
      if (!mounted) return;
      setState(() {
        _recording = false;
        _stopping = false;
        _elapsed = elapsed;
      });
      if (recorded == null) {
        setState(() => _error = 'The recording could not be saved.');
        return;
      }
      if (recorded.duration.inMilliseconds < 2500) {
        _showMessage('Record at least 3 seconds for a reliable analysis.');
      }
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsPage.analyze(
            input: AudioInput(filename: recorded.filename, path: recorded.path),
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _stopping = false;
        _recording = false;
        _error = 'Could not finish the recording: $error';
      });
    }
  }

  Future<void> _handleClose() async {
    if (_recording) {
      final shouldDiscard = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Discard recording?'),
          content: const Text('The current recording will be lost.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Keep recording'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Discard'),
            ),
          ],
        ),
      );
      if (shouldDiscard != true) return;
      await _recorder.cancel();
    }
    if (mounted) Navigator.of(context).pop();
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final busy = _starting || _stopping;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Record'),
        leading: IconButton(
          onPressed: _handleClose,
          icon: const Icon(Icons.close_rounded),
          tooltip: 'Close',
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
          child: Column(
            children: [
              const SizedBox(height: 12),
              Text(
                _recording
                    ? 'Listening...'
                    : 'Point your phone toward the birds',
                textAlign: TextAlign.center,
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w900,
                  letterSpacing: -0.4,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                _recording ? 'Tap the button again to stop.' : 'A clear 10-30 second recording works best. BirdEar needs at least 3 seconds of audio.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                  height: 1.45,
                ),
              ),
              const Spacer(),
              _RecordButton(
                recording: _recording,
                busy: busy,
                onTap: _toggleRecording,
              ),
              const SizedBox(height: 28),
              Text(
                formatClock(_elapsed.inMilliseconds / 1000),
                style: theme.textTheme.displaySmall?.copyWith(
                  fontWeight: FontWeight.w800,
                  fontFeatures: const [FontFeature.tabularFigures()],
                  color: _recording ? scheme.error : scheme.onSurface,
                ),
              ),
              Text(
                _recording ? 'recording' : 'ready',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                  letterSpacing: 0.6,
                ),
              ),
              const Spacer(),
              if (_error != null) ...[
                InfoBanner(
                  icon: Icons.error_outline_rounded,
                  color: scheme.error,
                  message: _error!,
                ),
                const SizedBox(height: 16),
              ],
              InfoBanner(
                message:
                    'Audio is sent to ${SettingsService.instance.baseUrl} for analysis.',
              ),
              const SizedBox(height: 12),
              Text(
                'Tip: avoid wind, traffic, and talking close to the microphone.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RecordButton extends StatelessWidget {
  const _RecordButton({
    required this.recording,
    required this.busy,
    required this.onTap,
  });

  final bool recording;
  final bool busy;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = recording ? scheme.error : scheme.primary;
    final foreground = recording ? scheme.onError : scheme.onPrimary;

    return GestureDetector(
      onTap: busy ? null : onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
        width: 184,
        height: 184,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: color.withValues(alpha: 0.12),
          boxShadow: [
            BoxShadow(
              color: color.withValues(alpha: recording ? 0.35 : 0.18),
              blurRadius: recording ? 48 : 24,
              spreadRadius: recording ? 12 : 4,
            ),
          ],
        ),
        alignment: Alignment.center,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          width: recording ? 112 : 132,
          height: recording ? 112 : 132,
          decoration: BoxDecoration(shape: BoxShape.circle, color: color),
          child: busy
              ? Padding(
                  padding: const EdgeInsets.all(44),
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: foreground,
                  ),
                )
              : Icon(
                  recording ? Icons.stop_rounded : Icons.mic_rounded,
                  size: recording ? 48 : 60,
                  color: foreground,
                ),
        ),
      ),
    );
  }
}
