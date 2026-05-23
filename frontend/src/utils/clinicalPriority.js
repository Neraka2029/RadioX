/**
 * Priorité clinique pour le résultat principal et les rapports.
 * TB et risque traumatique priment sur le simple score NIH maximal.
 */

export const TB_PRIMARY_THRESHOLD = 0.55;
export const FRACTURE_PRIMARY_THRESHOLD = 0.35;
export const CLINICAL_MIN_DISPLAY_SCORE = 0.05;
export const CONTRIBUTOR_MIN_SCORE = 0.05;
export const MAX_CONTRIBUTORS = 5;

const TRAUMA_RISK_LABEL = "Risque fracture costale";

const EXCLUDED_NIH_PRIMARY = new Set([
  "Normal",
  "Tuberculose",
  "Fracture costale",
  TRAUMA_RISK_LABEL,
  "Fracture costale suspectée",
]);

const ADDITIONAL_PATHOLOGY_KEYS = [
  {
    pathology: "Tuberculose",
    scoreKeys: [
      "tuberculose_score",
      "tuberculosis_display_score",
      "tuberculosis_probability",
    ],
  },
];

const TB_CONTRIBUTOR_HINTS = [
  "Infiltration",
  "Consolidation",
  "Nodule",
  "Epanchement pleural",
  "Fibrose",
  "Epaississement pleural",
];

const TRAUMA_CONTRIBUTOR_HINTS = [
  "Pneumothorax",
  "Atelectasie",
  "Fibrose",
  "Infiltration",
  "Consolidation",
  "Epanchement pleural",
];

function severityFromProbability(p) {
  if (p >= 0.65) return "high";
  if (p >= 0.5) return "moderate";
  return "low";
}

/**
 * Fusionne predictions[] avec scores TB / fracture pour l'affichage.
 */
export function mergeDisplayPredictions(results) {
  const base = [...(results?.predictions || [])];
  const indexByName = new Map(base.map((p, i) => [p.pathology, i]));

  for (const entry of ADDITIONAL_PATHOLOGY_KEYS) {
    const { pathology, scoreKeys, onlyWhen } = entry;
    if (onlyWhen && !onlyWhen(results)) {
      continue;
    }

    const fromList = base[indexByName.get(pathology)];
    const fromScore = scoreKeys
      .map((k) => results?.[k])
      .find((v) => typeof v === "number");
    const probability =
      typeof fromScore === "number"
        ? fromScore
        : (fromList?.probability ?? 0);

    const merged = {
      pathology,
      probability,
      severity: fromList?.severity ?? severityFromProbability(probability),
      description: fromList?.description || "",
      color: fromList?.color,
    };

    if (indexByName.has(pathology)) {
      base[indexByName.get(pathology)] = { ...fromList, ...merged };
    } else if (probability > 0) {
      indexByName.set(pathology, base.length);
      base.push(merged);
    }
  }

  const fractureMode = results?.fracture_mode ?? "none";
  const fractureScore = Number(
    results?.fracture_risk_score ?? results?.fracture_score ?? 0
  );

  if (fractureMode !== "yolo") {
    const fcIndex = base.findIndex((p) => p.pathology === "Fracture costale");
    if (fractureScore > 0) {
      const existing = base.find((p) => p.pathology === TRAUMA_RISK_LABEL);
      const traumaEntry = {
        pathology: TRAUMA_RISK_LABEL,
        probability: fractureScore,
        severity: severityFromProbability(fractureScore),
        description:
          "Proxy de risque traumatique (signaux radiologiques associés) — non confirmatoire.",
        color: "#ff7f50",
        isClinicalDerived: true,
        ...(existing || {}),
      };
      if (existing) {
        Object.assign(existing, traumaEntry);
      } else {
        base.push(traumaEntry);
      }
    }
  }

  return base;
}

/**
 * Filtre l'affichage : TB/Fracture costale selon seuils et mode YOLO.
 */
export function filterDisplayPredictions(predictions, results = null) {
  const fractureMode = results?.fracture_mode ?? "none";

  return predictions.filter((item) => {

    // TB
    if (item.pathology === "Tuberculose") {
      return item.probability > CLINICAL_MIN_DISPLAY_SCORE;
    }

    // SUPPRIMER totalement fracture costale générique
    if (item.pathology === "Fracture costale") {
      return false;
    }

    // Affichage intelligent selon mode
    if (item.pathology === TRAUMA_RISK_LABEL) {
      return (
        fractureMode !== "yolo" &&
        item.probability > CLINICAL_MIN_DISPLAY_SCORE
      );
    }

    if (item.pathology === "Fracture costale suspectée") {
      return (
        fractureMode === "yolo" &&
        item.probability > CLINICAL_MIN_DISPLAY_SCORE
      );
    }

    return true;
  });
}

function getNihPredictions(displayPredictions) {
  return displayPredictions
    .filter((p) => !EXCLUDED_NIH_PRIMARY.has(p.pathology))
    .sort((a, b) => b.probability - a.probability);
}

function getHighestNihFinding(displayPredictions) {
  const nih = getNihPredictions(displayPredictions);
  if (nih.length === 0) {
    return null;
  }
  const top = nih[0];
  return {
    pathology: top.pathology,
    severity: top.severity ?? severityFromProbability(top.probability),
    confidence: top.probability,
    probability: top.probability,
    priority: 3,
    source: "nih",
  };
}

/**
 * Détermine le résultat principal selon la priorité clinique.
 */
export function resolveClinicalPrimaryFinding(results, displayPredictions) {
  const tbScore = Number(
    results?.tuberculosis_probability ?? results?.tuberculose_score ?? 0
  );
  const fractureScore = Number(
    results?.fracture_risk_score ?? results?.fracture_score ?? 0
  );
  const fractureMode = results?.fracture_mode ?? "none";

  if (tbScore >= TB_PRIMARY_THRESHOLD) {
    return {
      pathology: "Tuberculose",
      severity: "high",
      confidence: tbScore,
      probability: tbScore,
      priority: 1,
      source: "clinical-tb",
    };
  }

  if (fractureScore >= FRACTURE_PRIMARY_THRESHOLD) {
    if (fractureMode === "yolo" && (results?.fracture_detections?.length ?? 0) > 0) {
      return {
        pathology: "Fracture costale suspectée",
        severity: "moderate",
        confidence: fractureScore,
        probability: fractureScore,
        priority: 2,
        source: "clinical-fracture-yolo",
      };
    }
    return {
      pathology: "Risque fracture costale",
      severity: "moderate",
      confidence: fractureScore,
      probability: fractureScore,
      priority: 2,
      source: "clinical-trauma",
    };
  }

  return getHighestNihFinding(displayPredictions);
}

function pickContributors(displayPredictions, hints, excludePathology) {
  const nih = getNihPredictions(displayPredictions).filter(
    (p) => p.probability >= CONTRIBUTOR_MIN_SCORE && p.pathology !== excludePathology
  );

  const hinted = [];
  const rest = [];

  for (const p of nih) {
    if (hints.includes(p.pathology)) {
      hinted.push(p);
    } else {
      rest.push(p);
    }
  }

  hinted.sort((a, b) => {
    const ai = hints.indexOf(a.pathology);
    const bi = hints.indexOf(b.pathology);
    if (ai !== bi) return ai - bi;
    return b.probability - a.probability;
  });

  return [...hinted, ...rest].slice(0, MAX_CONTRIBUTORS);
}

/**
 * Normalise les champs d'analyse (alias API / historique / rapports).
 */
export function normalizeAnalysisResults(results) {
  if (!results) return null;

  return {
    ...results,
    predictions: results.predictions || [],
    tuberculosis_probability: Number(
      results.tuberculosis_probability ?? results.tuberculose_score ?? 0
    ),
    fracture_risk_score: Number(
      results.fracture_risk_score ?? results.fracture_score ?? 0
    ),
    fracture_mode: results.fracture_mode ?? "none",
    fracture_detections: results.fracture_detections ?? [],
  };
}

/**
 * Payload PDF / export — mêmes champs que l'écran résultats après /predict.
 */
export function buildPdfAnalysisPayload(results) {
  const normalized = normalizeAnalysisResults(results);
  if (!normalized) return null;

  return {
    ...normalized,
    patientId: normalized.patient_id ?? normalized.patientId,
    analysis_id: normalized.analysis_id,
    date: normalized.timestamp
      ? new Date(normalized.timestamp).toLocaleDateString("fr-FR")
      : normalized.date,
  };
}

export function buildPdfImagePayload(results, uploadedImageUrl = null) {
  let image_base64 = results?.processed_image_base64 ?? null;

  if (!image_base64 && uploadedImageUrl) {
    if (uploadedImageUrl.startsWith("data:")) {
      image_base64 = uploadedImageUrl.split(",")[1] ?? null;
    } else if (!uploadedImageUrl.includes("http")) {
      image_base64 = uploadedImageUrl;
    }
  }

  return {
    image_base64,
    heatmap_base64: results?.heatmap_base64 ?? null,
  };
}

export function getClinicalDisplayState(results) {
  const normalized = normalizeAnalysisResults(results);
  if (!normalized) {
    return {
      normalized: null,
      mergedPredictions: [],
      displayPredictions: [],
      clinical: { primary: null, conclusionLine: "", contributors: [] },
      primary: null,
      conclusionLine: "",
      contributors: [],
    };
  }

  const mergedPredictions = mergeDisplayPredictions(normalized);
  const displayPredictions = filterDisplayPredictions(
    mergedPredictions,
    normalized
  );
  const clinical = buildClinicalConclusion(normalized, displayPredictions);

  return {
    normalized,
    mergedPredictions,
    displayPredictions,
    clinical,
    primary: clinical.primary,
    conclusionLine: clinical.conclusionLine,
    contributors: clinical.contributors,
  };
}

/**
 * Conclusion structurée pour l'UI et le PDF.
 */
export function buildClinicalConclusion(results, displayPredictions) {
  const primary = resolveClinicalPrimaryFinding(results, displayPredictions);

  if (!primary) {
    return {
      primary: null,
      conclusionLine: "",
      contributors: [],
    };
  }

  const score = primary.confidence ?? primary.probability ?? 0;
  const pct = Math.round(score * 100);
  const conclusionLine = `${primary.pathology}`;

  let contributors = [];
  if (primary.priority === 1) {
    contributors = pickContributors(displayPredictions, TB_CONTRIBUTOR_HINTS);
  } else if (primary.priority === 2) {
    contributors = pickContributors(displayPredictions, TRAUMA_CONTRIBUTOR_HINTS);
  } else {
    contributors = getNihPredictions(displayPredictions)
      .filter(
        (p) =>
          p.pathology !== primary.pathology &&
          p.probability >= CONTRIBUTOR_MIN_SCORE
      )
      .slice(0, MAX_CONTRIBUTORS);
  }

  return {
    primary,
    conclusionLine,
    contributors,
  };
}

/**
 * Seuils d'affichage pour la carte « Conclusion clinique » (secondaire).
 */
export function shouldShowClinicalConclusionCard(results) {
  const tb = Number(
    results?.tuberculosis_probability ?? results?.tuberculose_score ?? 0
  );
  const fracture = Number(
    results?.fracture_risk_score ?? results?.fracture_score ?? 0
  );
  return (
    tb > CLINICAL_MIN_DISPLAY_SCORE ||
    fracture > CLINICAL_MIN_DISPLAY_SCORE
  );
}

export function getClinicalConclusionLines(results) {
  const lines = [];
  const tb = Number(
    results?.tuberculosis_probability ?? results?.tuberculose_score ?? 0
  );
  const fracture = Number(
    results?.fracture_risk_score ?? results?.fracture_score ?? 0
  );

  if (tb > CLINICAL_MIN_DISPLAY_SCORE) {
    lines.push({
      label: "Tuberculose suspectée",
      score: tb,
    });
  }

  if (fracture > CLINICAL_MIN_DISPLAY_SCORE) {
    const label =
      results?.fracture_mode === "yolo"
        ? "Fracture costale suspectée"
        : "Risque traumatique thoracique";
    lines.push({ label, score: fracture });
  }

  return lines;
}
