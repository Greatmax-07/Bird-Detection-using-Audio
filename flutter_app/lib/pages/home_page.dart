import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../models/audio_input.dart';
import '../models/detection_result.dart';
import '../models/health_status.dart';
import '../services/api_service.dart';
import '../services/history_service.dart';
import '../services/settings_service.dart';
import '../utils/formatters.dart';
import '../widgets/empty_state.dart';
import '../widgets/info_banner.dart';
import '../widgets/section_header.dart';
import '../widgets/status_chip.dart';
import 'record_page.dart';
import 'results_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({
    super.key,
    required this.onOpenHistory,
    required this.onOpenSettings,
  });

  final VoidCallback onOpenHistory;
  final VoidCallback onOpenSettings;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ApiService _api = ApiService();

  HealthStatus? _health;
  String? _healthError;
  bool _checking = true;
  bool _picking = false;

  @override
  void initState() {
    super.initState();
    SettingsService.instance.addListener(_onSettingsChanged);
    _checkHealth();
  }

  @override
  void dispose() {
    SettingsService.instance.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (!mounted) return;
    setState(() {});
    _checkHealth();
  }

  Future<void> _checkHealth() async {
    if (mounted) {
      setState(() {
        _checking = true;
        _healthError = null;
      });
    }
    try {
      final health = await _api.health();
      if (!mounted) return;
      setState(() {
        _health = health;
        _checking = false;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _health = null;
        _healthError = error.message;
        _checking = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _health = null;
        _healthError = '$error';
        _checking = false;
      });
    }
  }

  Future<void> _pickAudio() async {
    if (_picking) return;
    setState(() => _picking = true);
    try {
      final picked = await FilePicker.pickFile(
        dialogTitle: 'Choose a bird recording',
        type: FileType.custom,
        allowedExtensions: const ['wav', 'mp3', 'flac', 'ogg', 'm4a'],
      );
      if (picked == null) return;
      final bytes = await picked.readAsBytes();
      if (bytes.isEmpty) {
        _showMessage('That audio file is empty.');
        return;
      }
      if (bytes.length > 30 * 1024 * 1024) {
        _showMessage('Audio must be 30 MiB or smaller.');
        return;
      }
      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsPage.analyze(
            input: AudioInput(filename: picked.name, bytes: bytes),
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      _showMessage('Could not read that file: $error');
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  void _openRecord() {
    Navigator.of(context)
        .push(MaterialPageRoute(builder: (_) => const RecordPage()));
  }

  void _openResult(DetectionResult result) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ResultsPage.fromResult(initialResult: result),
      ),
    );
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _checkHealth,
          child: ListView(
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 28),
            children: [
              _Header(online: _health != null, checking: _checking),
              const SizedBox(height: 18),
              const _HeroCard(),
              const SizedBox(height: 18),
              _ActionRow(
                busy: _picking,
                onRecord: _openRecord,
                onUpload: _pickAudio,
              ),
              const SizedBox(height: 18),
              _ServerCard(
                health: _health,
                error: _healthError,
                checking: _checking,
                onRetry: _checkHealth,
                onOpenSettings: widget.onOpenSettings,
              ),
              const SizedBox(height: 26),
              AnimatedBuilder(
                animation: HistoryService.instance,
                builder: (context, _) {
                  final entries = HistoryService.instance.entries
                      .take(3)
                      .toList();
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SectionHeader(
                        title: 'Recent detections',
                        actionLabel: entries.isEmpty ? null : 'View all',
                        onAction: widget.onOpenHistory,
                      ),
                      const SizedBox(height: 10),
                      if (entries.isEmpty)
                        Card(
                          child: EmptyState(
                            icon: Icons.history_rounded,
                            title: 'No detections yet',
                            message: 'Record a clip or upload an audio file to start identifying birds around you.',
                            actionLabel: 'Record now',
                            onAction: _openRecord,
                          ),
                        )
                      else
                        for (final entry in entries)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 10),
                            child: _RecentTile(
                              result: entry,
                              onTap: () => _openResult(entry),
                            ),
                          ),
                    ],
                  );
                },
              ),
              const SizedBox(height: 20),
              const InfoBanner(
                message: 'Counts describe separate stretches of calling, not the number of individual birds.',
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.online, required this.checking});

  final bool online;
  final bool checking;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Row(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: scheme.primaryContainer,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Icon(
            Icons.forest_rounded,
            color: scheme.onPrimaryContainer,
            size: 26,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'BirdEar',
                style: theme.textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w900,
                  letterSpacing: -0.4,
                ),
              ),
              Text(
                'Maharashtra bird identification',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
        StatusChip(online: online, loading: checking),
      ],
    );
  }
}

class _HeroCard extends StatelessWidget {
  const _HeroCard();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: LinearGradient(
          colors: [scheme.primary, scheme.tertiary],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            right: -18,
            top: -28,
            child: Icon(
              Icons.forest_rounded,
              size: 168,
              color: Colors.white.withValues(alpha: 0.13),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Listen. Identify. Discover.',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.w900,
                    height: 1.2,
                    letterSpacing: -0.4,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Detect the birds of Maharashtra from any recording, with timestamps for every call.',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.9),
                    height: 1.45,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 16),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: const [
                    _HeroTag(label: '68 species'),
                    _HeroTag(label: 'Timestamps'),
                    _HeroTag(label: 'Multi-species'),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroTag extends StatelessWidget {
  const _HeroTag({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withValues(alpha: 0.22)),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }
}

class _ActionRow extends StatelessWidget {
  const _ActionRow({
    required this.busy,
    required this.onRecord,
    required this.onUpload,
  });

  final bool busy;
  final VoidCallback onRecord;
  final VoidCallback onUpload;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: onRecord,
            icon: const Icon(Icons.mic_rounded),
            label: const Text('Record'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: busy ? null : onUpload,
            icon: busy
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.upload_file_rounded),
            label: Text(busy ? 'Opening' : 'Upload'),
          ),
        ),
      ],
    );
  }
}

class _ServerCard extends StatelessWidget {
  const _ServerCard({
    required this.health,
    required this.error,
    required this.checking,
    required this.onRetry,
    required this.onOpenSettings,
  });

  final HealthStatus? health;
  final String? error;
  final bool checking;
  final VoidCallback onRetry;
  final VoidCallback onOpenSettings;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.dns_outlined, size: 20, color: scheme.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'BirdEar server',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
                TextButton(
                  onPressed: onOpenSettings,
                  child: const Text('Settings'),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              SettingsService.instance.baseUrl,
              style: theme.textTheme.bodySmall?.copyWith(
                color: scheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            if (checking) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: const LinearProgressIndicator(minHeight: 6),
              ),
              const SizedBox(height: 10),
              Text(
                'Checking server connection...',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ] else if (error != null) ...[
              Text(
                error!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.error,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 8),
              TextButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Retry connection'),
              ),
            ] else if (health != null) ...[
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  _StatChip(label: '${health!.nClasses} species'),
                  _StatChip(
                    label: '${health!.nWithBirdnetScores} BirdNET scores',
                  ),
                  if (health!.params['w_birdnet'] != null)
                    _StatChip(
                      label: 'w ${_shortNumber(health!.params['w_birdnet'])}',
                    ),
                  if (health!.params['t_high'] != null)
                    _StatChip(
                      label: 't_high ${_shortNumber(health!.params['t_high'])}',
                    ),
                  if (health!.params['t_low'] != null)
                    _StatChip(
                      label: 't_low ${_shortNumber(health!.params['t_low'])}',
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  static String _shortNumber(dynamic value) {
    if (value is num) {
      if (value == value.roundToDouble()) return value.toInt().toString();
      return value.toStringAsFixed(2);
    }
    return value.toString();
  }
}

class _StatChip extends StatelessWidget {
  const _StatChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: scheme.primaryContainer.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: scheme.onPrimaryContainer,
          fontSize: 12,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }
}

class _RecentTile extends StatelessWidget {
  const _RecentTile({required this.result, required this.onTap});

  final DetectionResult result;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final top = result.speciesDetected.isEmpty
        ? null
        : result.speciesDetected.first;
    final title = top?.species ?? 'No confident detection';
    final subtitle = top == null
        ? formatShortDate(result.analyzedAt)
        : '${result.speciesDetected.length} species · ${formatShortDate(result.analyzedAt)}';

    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: top == null
                      ? scheme.surfaceContainerHighest
                      : scheme.primaryContainer,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  top == null
                      ? Icons.search_off_rounded
                      : Icons.graphic_eq_rounded,
                  color: top == null
                      ? scheme.onSurfaceVariant
                      : scheme.onPrimaryContainer,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.bodyLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              if (top != null)
                Text(
                  formatPercent(top.confidence),
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w800,
                    color: scheme.primary,
                  ),
                ),
              const SizedBox(width: 4),
              Icon(Icons.chevron_right_rounded, color: scheme.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}
