/**
 * Bayesian Diagnostic Engine
 * Implements formal Bayesian probability updating for medical diagnoses
 */

export interface DiagnosticEvidence {
  type: 'symptom' | 'test_result' | 'demographic' | 'history';
  name: string;
  value: string | number | boolean;
  confidence: number; // 0-1
  timestamp: Date;
}

export interface BayesianDiagnosis {
  condition: string;
  priorProbability: number;
  posteriorProbability: number;
  likelihood: number;
  evidence: DiagnosticEvidence[];
  icd10Code?: string;
}

export interface BayesianUpdate {
  diagnosis: string;
  oldProbability: number;
  newProbability: number;
  evidence: DiagnosticEvidence;
  likelihoodRatio: number;
}

export class BayesianDiagnosticEngine {
  private diagnoses: Map<string, BayesianDiagnosis> = new Map();
  private evidenceBase: Map<string, Map<string, number>> = new Map(); // condition -> evidence -> likelihood

  constructor() {
    this.initializeEvidenceBase();
  }

  /**
   * Initialize the evidence base with medical knowledge
   */
  private initializeEvidenceBase(): void {
    // Myocardial Infarction evidence patterns
    this.setLikelihood('myocardial_infarction', 'chest_pain_crushing', 8.0);  // High LR for MI
    this.setLikelihood('myocardial_infarction', 'diaphoresis', 3.0);
    this.setLikelihood('myocardial_infarction', 'nausea', 2.0);
    this.setLikelihood('myocardial_infarction', 'troponin_elevated', 20.0);  // Very high LR
    this.setLikelihood('myocardial_infarction', 'age_over_65', 2.5);
    this.setLikelihood('myocardial_infarction', 'fever', 0.3);  // Fever uncommon in MI

    // Pneumonia evidence patterns
    this.setLikelihood('pneumonia', 'fever', 4.0);
    this.setLikelihood('pneumonia', 'no_fever', 0.3);  // Absence of fever reduces pneumonia likelihood
    this.setLikelihood('pneumonia', 'cough', 5.0);
    this.setLikelihood('pneumonia', 'shortness_of_breath', 3.0);
    this.setLikelihood('pneumonia', 'chest_xray_infiltrate', 15.0);
    this.setLikelihood('pneumonia', 'chest_pain_crushing', 0.2);  // Less common
    this.setLikelihood('pneumonia', 'troponin_elevated', 0.1);  // Very rare

    // Gastroenteritis evidence patterns
    this.setLikelihood('gastroenteritis', 'nausea', 4.0);
    this.setLikelihood('gastroenteritis', 'vomiting', 6.0);
    this.setLikelihood('gastroenteritis', 'diarrhea', 8.0);
    this.setLikelihood('gastroenteritis', 'abdominal_pain', 5.0);
    this.setLikelihood('gastroenteritis', 'fever', 2.0);  // Sometimes present
    this.setLikelihood('gastroenteritis', 'chest_pain_crushing', 0.1);  // Very rare
    this.setLikelihood('gastroenteritis', 'troponin_elevated', 0.05);  // Extremely rare
  }

  /**
   * Set likelihood of evidence for a specific condition
   */
  private setLikelihood(condition: string, evidence: string, likelihood: number): void {
    if (!this.evidenceBase.has(condition)) {
      this.evidenceBase.set(condition, new Map());
    }
    this.evidenceBase.get(condition)!.set(evidence, likelihood);
  }

  /**
   * Initialize diagnosis with prior probability based on prevalence
   */
  initializeDiagnosis(condition: string, priorProbability: number, icd10Code?: string): void {
    this.diagnoses.set(condition, {
      condition,
      priorProbability,
      posteriorProbability: priorProbability,
      likelihood: 1.0,
      evidence: [],
      icd10Code
    });
  }

  /**
   * Update diagnosis probabilities based on new evidence using Bayes' theorem
   */
  updateWithEvidence(evidence: DiagnosticEvidence): BayesianUpdate[] {
    const updates: BayesianUpdate[] = [];

    for (const [condition, diagnosis] of Array.from(this.diagnoses.entries())) {
      const likelihoodRatio = this.calculateLikelihoodRatio(condition, evidence);
      const oldProbability = diagnosis.posteriorProbability;
      
      // Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
      // Using odds form: odds(H|E) = LR * odds(H)
      const oldOdds = oldProbability / (1 - oldProbability);
      const newOdds = likelihoodRatio * oldOdds;
      const newProbability = newOdds / (1 + newOdds);

      // Update diagnosis
      diagnosis.posteriorProbability = Math.min(Math.max(newProbability, 0.001), 0.999);
      diagnosis.evidence.push(evidence);
      diagnosis.likelihood *= likelihoodRatio;

      updates.push({
        diagnosis: condition,
        oldProbability,
        newProbability: diagnosis.posteriorProbability,
        evidence,
        likelihoodRatio
      });
    }

    // Normalize probabilities to ensure they sum to reasonable total
    this.normalizeProbabilities();

    return updates;
  }

  /**
   * Calculate likelihood ratio for evidence given condition
   */
  private calculateLikelihoodRatio(condition: string, evidence: DiagnosticEvidence): number {
    const evidenceKey = this.generateEvidenceKey(evidence);
    const conditionEvidence = this.evidenceBase.get(condition);
    
    if (!conditionEvidence || !conditionEvidence.has(evidenceKey)) {
      // Default likelihood for unknown evidence - slight towards neutral
      return 1.0;
    }

    const likelihoodRatio = conditionEvidence.get(evidenceKey)!;
    
    // Apply confidence adjustment: LR_adjusted = 1 + (LR - 1) * confidence
    // This works for both positive (LR > 1) and negative (LR < 1) evidence
    return 1 + (likelihoodRatio - 1) * evidence.confidence;
  }

  /**
   * Generate standardized evidence key
   */
  private generateEvidenceKey(evidence: DiagnosticEvidence): string {
    // Normalize evidence name and value into a standardized key
    const normalizedName = evidence.name.toLowerCase().replace(/\s+/g, '_');
    const normalizedValue = String(evidence.value).toLowerCase();
    
    // For boolean values
    if (typeof evidence.value === 'boolean') {
      return evidence.value ? normalizedName : `no_${normalizedName}`;
    }
    
    // For specific values
    if (normalizedValue === 'true' || normalizedValue === 'positive' || normalizedValue === 'elevated') {
      return normalizedName;
    }
    
    // For numeric values (could be enhanced with ranges)
    if (typeof evidence.value === 'number') {
      return `${normalizedName}_${normalizedValue}`;
    }
    
    return `${normalizedName}_${normalizedValue}`;
  }


  /**
   * Normalize probabilities to prevent extreme values
   */
  private normalizeProbabilities(): void {
    // Simple normalization to keep probabilities reasonable
    // In a more sophisticated implementation, this would ensure probabilities
    // for mutually exclusive conditions sum appropriately
    
    const totalProbability = Array.from(this.diagnoses.values())
      .reduce((sum, diagnosis) => sum + diagnosis.posteriorProbability, 0);
    
    if (totalProbability > 1.0) {
      const normalizationFactor = 0.95 / totalProbability; // Leave some room for uncertainty
      
      for (const diagnosis of Array.from(this.diagnoses.values())) {
        diagnosis.posteriorProbability *= normalizationFactor;
      }
    }
  }

  /**
   * Get all diagnoses sorted by probability
   */
  getRankedDiagnoses(): BayesianDiagnosis[] {
    return Array.from(this.diagnoses.values())
      .sort((a, b) => b.posteriorProbability - a.posteriorProbability);
  }

  /**
   * Get diagnosis by condition name
   */
  getDiagnosis(condition: string): BayesianDiagnosis | undefined {
    return this.diagnoses.get(condition);
  }

  /**
   * Calculate information gain for potential evidence
   */
  calculateInformationGain(potentialEvidence: Omit<DiagnosticEvidence, 'timestamp'>): number {
    // Calculate expected information gain using entropy reduction
    let currentEntropy = this.calculateEntropy();
    let expectedEntropyAfterEvidence = 0;

    // For each possible outcome of the evidence
    const outcomes = [true, false]; // Simplified binary outcomes
    
    for (const outcome of outcomes) {
      const tempEvidence: DiagnosticEvidence = {
        ...potentialEvidence,
        value: outcome,
        timestamp: new Date()
      };

      // Calculate probabilities after this evidence
      const hypotheticalUpdates = this.calculateHypotheticalUpdates(tempEvidence);
      
      // Calculate entropy after this evidence
      const entropyAfterEvidence = this.calculateEntropyFromUpdates(hypotheticalUpdates);
      
      // Weight by probability of this outcome
      const outcomeWeight = this.calculateOutcomeProbability(tempEvidence);
      expectedEntropyAfterEvidence += outcomeWeight * entropyAfterEvidence;
    }

    return currentEntropy - expectedEntropyAfterEvidence;
  }

  /**
   * Calculate current entropy of diagnosis probabilities
   */
  private calculateEntropy(): number {
    let entropy = 0;
    const probabilities = Array.from(this.diagnoses.values()).map(d => d.posteriorProbability);
    const total = probabilities.reduce((sum, p) => sum + p, 0);

    for (const prob of probabilities) {
      if (prob > 0) {
        const normalizedProb = prob / total;
        entropy -= normalizedProb * Math.log2(normalizedProb);
      }
    }

    return entropy;
  }

  /**
   * Calculate hypothetical updates without actually applying them
   */
  private calculateHypotheticalUpdates(evidence: DiagnosticEvidence): BayesianUpdate[] {
    const updates: BayesianUpdate[] = [];

    for (const [condition, diagnosis] of Array.from(this.diagnoses.entries())) {
      const likelihoodRatio = this.calculateLikelihoodRatio(condition, evidence);
      const oldProbability = diagnosis.posteriorProbability;
      
      const oldOdds = oldProbability / (1 - oldProbability);
      const newOdds = likelihoodRatio * oldOdds;
      const newProbability = newOdds / (1 + newOdds);

      updates.push({
        diagnosis: condition,
        oldProbability,
        newProbability: Math.min(Math.max(newProbability, 0.001), 0.999),
        evidence,
        likelihoodRatio
      });
    }

    return updates;
  }

  /**
   * Calculate entropy from hypothetical updates
   */
  private calculateEntropyFromUpdates(updates: BayesianUpdate[]): number {
    let entropy = 0;
    const probabilities = updates.map(u => u.newProbability);
    const total = probabilities.reduce((sum, p) => sum + p, 0);

    for (const prob of probabilities) {
      if (prob > 0) {
        const normalizedProb = prob / total;
        entropy -= normalizedProb * Math.log2(normalizedProb);
      }
    }

    return entropy;
  }

  /**
   * Calculate probability of evidence outcome (simplified)
   */
  private calculateOutcomeProbability(_evidence: DiagnosticEvidence): number {
    // Simplified: assume 50/50 for binary outcomes
    // In practice, this would be based on population statistics
    return 0.5;
  }

  /**
   * Reset engine for new case
   */
  reset(): void {
    this.diagnoses.clear();
  }
}