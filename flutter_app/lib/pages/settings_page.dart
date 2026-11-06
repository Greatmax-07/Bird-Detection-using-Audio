import 'package:flutter/material.dart';

import '../models/health_status.dart';
import '../services/api_service.dart';
import '../services/history_service.dart';
import '../services/settings_service.dart';
import '../widgets/info_banner.dart';
import '../widgets/section_header.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final TextEditingController _controller;
  bool _saving = false;
  bool _testing = false;
  HealthStatus? _health;
  String? _testError;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: SettingsService.instance.baseUrl);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final value = _controller.text.trim();
    final validationError = _validateUrl(value);
    if (validationError != null) {
      _showMessage(validationError);
      return;
    }
    setState(() => _saving = true);
    await SettingsService.instance.updateBaseUrl(value);
    if (!mounted) return;
    setState(() => _saving = false);
    _showMessage('Server URL saved.');
    await _testConnection();
  }

  Future<void> _testConnection() async {
    final value = _controller.text.trim();
    final validationError = _validateUrl(value);
    if (validationError != null) {
      setState(() {
        _testError = validationError;
        _health = null;
      });
      return;
    }

    setState(() {
      _testing = true;
      _testError = null;
      _health = null;
    });
    try {
      final health = await ApiService(baseUrl: value).health();
      if (!mounted) return;
      setState(() {
        _health = health;
        _testing = false;
      });
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _testError = error.message;
        _testing = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _testError = '$error';
        _testing = false;
      });
    }
  }

  void _resetUrl() {
    _controller.text = SettingsService.defaultBaseUrl();
    _save();
  }

  Future<void> _confirmClearHistory() async {
    final shouldClear = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear history?'),
        content: const Text(
          'This removes every saved detection from this device.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (shouldClear == true) {
      await HistoryService.instance.clear();
      if (!mounted) return;
      _showMessage('Detection history cleared.');
    }
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  String? _validateUrl(String value) {
    final uri = Uri.tryParse(value);
    if (uri == null ||
        !uri.hasScheme ||
        (uri.scheme != 'http' && uri.scheme != 'https') ||
        uri.host.isEmpty) {
      return 'Enter a full URL, for example http://192.168.1.5:8000';
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;

    return AnimatedBuilder(
      animation: HistoryService.instance,
      builder: (context, _) {
        final historyCount = HistoryService.instance.entries.length;
        return Scaffold(
          appBar: AppBar(title: const Text('Settings')),
          body: ListView(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
            children: [
              const SectionHeader(
                title: 'Server connection',
                subtitle: 'Point BirdEar at your FastAPI backend.',
              ),
              const SizedBox(height: 10),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      TextField(
                        controller: _controller,
                        keyboardType: TextInputType.url,
                        autocorrect: false,
                        textInputAction: TextInputAction.done,
                        onSubmitted: (_) => _save(),
                        decoration: const InputDecoration(
                          labelText: 'BirdEar API URL',
                          hintText: 'http://10.0.2.2:8000',
                          prefixIcon: Icon(Icons.link_rounded),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        'Android emulator: http://10.0.2.2:8000\n'
                        'Physical device: use your computer\'s LAN IP.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: scheme.onSurfaceVariant,
                          height: 1.45,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Row(
                        children: [
                          Expanded(
                            child: FilledButton.icon(
                              onPressed: _saving ? null : _save,
                              icon: _saving
                                  ? const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.save_outlined),
                              label: Text(_saving ? 'Saving' : 'Save'),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: _testing ? null : _testConnection,
                              icon: _testing
                                  ? const SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.wifi_tethering_rounded),
                              label: Text(_testing ? 'Testing' : 'Test'),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: TextButton(
                          onPressed: _resetUrl,
                          child: const Text('Reset to default'),
                        ),
                      ),
                      if (_testing) ...[
                        const SizedBox(height: 8),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(999),
                          child: const LinearProgressIndicator(minHeight: 5),
                        ),
                      ],
                      if (_testError != null) ...[
                        const SizedBox(height: 12),
                        InfoBanner(
                          icon: Icons.error_outline_rounded,
                          color: scheme.error,
                          message: _testError!,
                        ),
                      ],
                      if (_health != null) ...[
                        const SizedBox(height: 12),
                        _ConnectionDetails(health: _health!),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 26),
              const SectionHeader(
                title: 'Data',
                subtitle: 'Stored locally on this device.',
              ),
              const SizedBox(height: 10),
              Card(
                child: ListTile(
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 18,
                    vertical: 6,
                  ),
                  leading: Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                      color: scheme.primaryContainer,
                      borderRadius: BorderRadius.circular(13),
                    ),
                    child: Icon(
                      Icons.history_rounded,
                      color: scheme.onPrimaryContainer,
                      size: 21,
                    ),
                  ),
                  title: const Text(
                    'Detection history',
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  subtitle: Text(
                    '$historyCount saved ${historyCount == 1 ? 'detection' : 'detections'}',
                  ),
                  trailing: TextButton(
                    onPressed: historyCount == 0 ? null : _confirmClearHistory,
                    child: const Text('Clear'),
                  ),
                ),
              ),
              const SizedBox(height: 26),
              const SectionHeader(title: 'About BirdEar'),
              const SizedBox(height: 10),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    children: const [
                      _AboutRow(
                        icon: Icons.apps_rounded,
                        label: 'Version',
                        value: '1.0.0',
                      ),
                      SizedBox(height: 12),
                      _AboutRow(
                        icon: Icons.location_on_outlined,
                        label: 'Region',
                        value: 'Maharashtra, India',
                      ),
                      SizedBox(height: 12),
                      _AboutRow(
                        icon: Icons.forest_rounded,
                        label: 'Species scope',
                        value: '68 target species',
                      ),
                      SizedBox(height: 12),
                      _AboutRow(
                        icon: Icons.memory_rounded,
                        label: 'Model',
                        value: 'BirdNET + regional segment head',
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 26),
              const SectionHeader(
                title: 'Running the backend',
                subtitle: 'Start the FastAPI server from the project root.',
              ),
              const SizedBox(height: 10),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: scheme.surfaceContainerHighest.withValues(
                        alpha: 0.45,
                      ),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const SelectableText(
                      'uvicorn main:app --app-dir web/app --reload',
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 13,
                        height: 1.5,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              const InfoBanner(
                message: 'The server needs features/segment_head.pkl, features/label_map.json, and features/model_meta.json. Supported uploads: .wav, .mp3, .flac, .ogg, and .m4a up to 30 MiB.',
              ),
              const SizedBox(height: 12),
              Text(
                'BirdEar uses BirdNET v2.4 embeddings plus a regional classifier head trained on Maharashtra species. '
                'Counts are separate stretches of calling, not individual birds.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                  height: 1.45,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ConnectionDetails extends StatelessWidget {
  const _ConnectionDetails({required this.health});

  final HealthStatus health;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF2E7D32).withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.check_circle_rounded,
                color: Color(0xFF2E7D32),
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                'Connected',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: const Color(0xFF2E7D32),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '${health.nClasses} classes · '
            '${health.nWithBirdnetScores} with BirdNET scores',
            style: theme.textTheme.bodySmall?.copyWith(
              color: scheme.onSurfaceVariant,
            ),
          ),
          if (health.params.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              health.params.entries
                  .map((entry) => '${entry.key}: ${_formatValue(entry.value)}')
                  .join('  ·  '),
              style: theme.textTheme.bodySmall?.copyWith(
                color: scheme.onSurfaceVariant,
                height: 1.4,
              ),
            ),
          ],
        ],
      ),
    );
  }

  static String _formatValue(dynamic value) {
    if (value is num) {
      if (value == value.roundToDouble()) return value.toInt().toString();
      return value.toStringAsFixed(2);
    }
    return value.toString();
  }
}

class _AboutRow extends StatelessWidget {
  const _AboutRow({
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
      children: [
        Icon(icon, size: 19, color: scheme.onSurfaceVariant),
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
            textAlign: TextAlign.right,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}
