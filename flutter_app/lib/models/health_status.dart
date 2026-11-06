class HealthStatus {
  const HealthStatus({
    required this.status,
    required this.nClasses,
    required this.nWithBirdnetScores,
    required this.params,
  });

  final String status;
  final int nClasses;
  final int nWithBirdnetScores;
  final Map<String, dynamic> params;

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    final rawParams = json['params'];
    return HealthStatus(
      status: json['status']?.toString() ?? 'unknown',
      nClasses: _asInt(json['n_classes']),
      nWithBirdnetScores: _asInt(json['n_with_birdnet_scores']),
      params: rawParams is Map
          ? Map<String, dynamic>.from(rawParams)
          : const <String, dynamic>{},
    );
  }
}

int _asInt(dynamic value, [int fallback = 0]) {
  if (value is num) return value.toInt();
  return int.tryParse('$value') ?? fallback;
}
