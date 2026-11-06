import 'package:flutter/material.dart';

import 'pages/root_shell.dart';
import 'services/history_service.dart';
import 'services/settings_service.dart';
import 'theme/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SettingsService.instance.load();
  await HistoryService.instance.load();
  runApp(const BirdEarApp());
}

class BirdEarApp extends StatelessWidget {
  const BirdEarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BirdEar',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      darkTheme: AppTheme.dark(),
      themeMode: ThemeMode.system,
      home: const RootShell(),
    );
  }
}
