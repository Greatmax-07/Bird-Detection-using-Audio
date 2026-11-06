import 'package:flutter/material.dart';

import '../models/audio_input.dart';
import '../models/detection_result.dart';
import '../services/api_service.dart';
import '../services/history_service.dart';
import '../utils/formatters.dart';
import '../widgets/audio_player_card.dart';
import '../widgets/error_view.dart';
import '../widgets/info_banner.dart';
import '../widgets/prediction_tile.dart';
import '../widgets/section_header.dart';
import '../widgets/species_card.dart';

class ResultsPage extends StatefulWidget {
  const ResultsPage.analyze({super.key, required this.input})
    : initialResult = null;

  const ResultsPage.fromResult({super.key, required this.initialResult})
    : input = null;

  final AudioInput? input;
  final DetectionResult? initialResult;

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  final ApiService _api = ApiService();

  DetectionResult? _result;
  String? _error;
  bool _loading = false;
  bool _saved = false;

  @override
  void initState() {
    super.initState();
    _result = widget.initialResult;
    if (_result != null) {
      _saved = true;
    } else {
      _run();
    }
  }

  Future<void> _run() async {
    final input = widget.input;
    if (input == null) return;
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });
    try {
      final result = await _api.predict(input);
      if (!mounted) return;
      setState(() {
        _result = result;
        _loading = false;
      });
      if (!_saved) {
        _saved = true;
        await HistoryService.instance.add(result);
      }
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = error.message;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = 'There was a problem analyzing this audio: $error';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final canRetry = widget.input != null;
    return Scaffold(
      appBar: AppBar(
        title: Text(_loading ? 'Analyzing' : 'Analysis'),
        actions: [
          if (canRetry && !_loading)
            IconButton(
              onPressed: _run,
              tooltip: 'Analyze again',
              icon: const Icon(Icons.refresh_rounded),
            ),
        ],
      ),
      body: _buildBody(canRetry),
    );
  }

  Widget _buildBody(bool canRetry) {
    if (_loading) {
      return _AnalysisLoading(
        sourceName: widget.input?.displayName ?? 'Audio clip',
      );
    }
    if (_error != null) {
      return ErrorView(
        title: 'Could not analyze audio',
        message: _error!,
        onRetry: canRetry ? _run : null,
      );
    }
    final result = _result;
    if (result == null) {
      return const ErrorView(message: 'No analysis result is available.');
    }
    return _ResultBody(result: result, input: widget.input);
  }
}

class _AnalysisLoading extends StatelessWidget {
  const _AnalysisLoading({required this.sourceName});

  final String sourceName;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return SafeArea(
      child: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 92,
                height: 92,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    Container(
                      width: 92,
                      height: 92,
                      decoration: BoxDecoration(
                        color: scheme.primaryContainer.withValues(alpha: 0.55),
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(
                      width: 68,
                      height: 68,
                      child: CircularProgressIndicator(strokeWidth: 3),
                    ),
                    Icon(
                      Icons.graphic_eq_rounded,
                      color: scheme.primary,
                      size: 30,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Listening for birds...',
                style: theme.textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                sourceName,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'BirdNET is running a single pass over the clip. This can take a few seconds.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                  height: 1.45,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultBody extends StatelessWidget {
  const _ResultBody({required this.result, required this.input});

  final DetectionResult result;
  final AudioInput? input;

  @override
  Widget build(BuildContext context) {
    final species = result.speciesDetected;
    final predictions = result.predictions;
    final playbackInput = input;

    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
      children: [
        _SummaryCard(result: result),
        if (playbackInput != null && playbackInput.hasData) ...[
          const SizedBox(height: 20),
          AudioPlayerCard(input: playbackInput, result: result),
        ],
        const SizedBox(height: 20),
        if (species.isEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      color: Theme.of(context)
                          .colorScheme
                          .surfaceContainerHighest,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.search_off_rounded,
                      size: 30,
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 14),
                  Text(
                    'No confident bird detected',
                    style: Theme.of(context).textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w800),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Try a clearer, longer, or quieter recording.',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
        ] else ...[
          SectionHeader(
            title: 'Detected species',
            subtitle:
                '${species.length} species · ${result.eventCount} call ${result.eventCount == 1 ? 'event' : 'events'}',
          ),
          const SizedBox(height: 10),
          for (var index = 0; index < species.length; index++)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: SpeciesCard(
                detection: species[index],
                rank: index + 1,
                durationSeconds: result.durationSeconds,
              ),
            ),
          const SizedBox(height: 8),
        ],
        if (predictions.isNotEmpty) ...[
          SectionHeader(
            title: result.hasDetections
                ? 'Strongest scores'
                : 'Closest matches',
            subtitle: 'Top model predictions across the whole clip',
          ),
          const SizedBox(height: 10),
          Card(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
              child: Column(
                children: [
                  for (var index = 0; index < predictions.length; index++)
                    PredictionTile(
                      prediction: predictions[index],
                      rank: index + 1,
                      muted: !result.hasDetections,
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
        ],
        if (result.countNote.isNotEmpty)
          InfoBanner(message: result.countNote)
        else
          const InfoBanner(
            message: 'Counts describe separate stretches of calling, not individual birds.',
          ),
        const SizedBox(height: 20),
        _MetadataCard(result: result),
        const SizedBox(height: 20),
        OutlinedButton.icon(
          onPressed: () => Navigator.of(context).pop(),
          icon: const Icon(Icons.check_rounded),
          label: const Text('Done'),
        ),
      ],
    );
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.result});

  final DetectionResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final top = result.speciesDetected.isEmpty
        ? null
        : result.speciesDetected.first;
    final title = top == null
        ? 'No confident bird detected'
        : '${result.speciesDetected.length} species detected';
    final subtitle = top == null
        ? 'The clip did not pass the confidence threshold.'
        : '${result.totalApproximateCount} approximate call ${result.totalApproximateCount == 1 ? 'occurrence' : 'occurrences'} across ${formatDuration(result.durationSeconds)}';

    return Container(
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(26),
        gradient: LinearGradient(
          colors: [scheme.primary, scheme.tertiary],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            right: -20,
            bottom: -30,
            child: Icon(
              top == null ? Icons.search_off_rounded : Icons.forest_rounded,
              size: 160,
              color: Colors.white.withValues(alpha: 0.12),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(22),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 46,
                      height: 46,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.18),
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Icon(
                        top == null
                            ? Icons.search_off_rounded
                            : Icons.check_circle_outline_rounded,
                        color: Colors.white,
                        size: 26,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        title,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w900,
                          height: 1.2,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                Text(
                  subtitle,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.9),
                    height: 1.4,
                  ),
                ),
                if (top != null) ...[
                  const SizedBox(height: 14),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _SummaryTag(label: top.species),
                      _SummaryTag(
                        label: '${formatPercent(top.confidence)} confidence',
                      ),
                      _SummaryTag(
                        label: formatDuration(result.durationSeconds),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SummaryTag extends StatelessWidget {
  const _SummaryTag({required this.label});

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

class _MetadataCard extends StatelessWidget {
  const _MetadataCard({required this.result});

  final DetectionResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Clip details',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 12),
            _MetaRow(
              icon: Icons.audio_file_outlined,
              label: 'Source',
              value: result.sourceName,
            ),
            const SizedBox(height: 10),
            _MetaRow(
              icon: Icons.timer_outlined,
              label: 'Analyzed length',
              value: formatDuration(result.durationSeconds),
            ),
            const SizedBox(height: 10),
            _MetaRow(
              icon: Icons.schedule_rounded,
              label: 'Analyzed at',
              value: formatTimestamp(result.analyzedAt),
            ),
          ],
        ),
      ),
    );
  }
}

class _MetaRow extends StatelessWidget {
  const _MetaRow({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: scheme.onSurfaceVariant),
        const SizedBox(width: 10),
        SizedBox(
          width: 110,
          child: Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}
