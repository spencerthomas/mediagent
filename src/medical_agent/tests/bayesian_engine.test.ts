/**
 * Bayesian Diagnostic Engine Tests
 * Comprehensive test suite for Bayesian probability calculations
 */

import { BayesianDiagnosticEngine, DiagnosticEvidence } from '../bayesian_engine';

describe('BayesianDiagnosticEngine', () => {
  let engine: BayesianDiagnosticEngine;

  beforeEach(() => {
    engine = new BayesianDiagnosticEngine();
  });

  afterEach(() => {
    engine.reset();
  });

  describe('Initialization', () => {
    test('should initialize with empty diagnoses', () => {
      const diagnoses = engine.getRankedDiagnoses();
      expect(diagnoses).toHaveLength(0);
    });

    test('should initialize diagnosis with prior probability', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02, 'I21.9');
      
      const diagnosis = engine.getDiagnosis('myocardial_infarction');
      expect(diagnosis).toBeDefined();
      expect(diagnosis!.priorProbability).toBe(0.02);
      expect(diagnosis!.posteriorProbability).toBe(0.02);
      expect(diagnosis!.icd10Code).toBe('I21.9');
    });

    test('should initialize multiple diagnoses', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      engine.initializeDiagnosis('gastroenteritis', 0.15);

      const diagnoses = engine.getRankedDiagnoses();
      expect(diagnoses).toHaveLength(3);
      
      // Should be sorted by probability (gastroenteritis highest)
      expect(diagnoses[0].condition).toBe('gastroenteritis');
      expect(diagnoses[1].condition).toBe('pneumonia');
      expect(diagnoses[2].condition).toBe('myocardial_infarction');
    });
  });

  describe('Evidence Processing', () => {
    beforeEach(() => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      engine.initializeDiagnosis('gastroenteritis', 0.15);
    });

    test('should update probabilities with positive evidence', () => {
      const evidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'chest_pain_crushing',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      const updates = engine.updateWithEvidence(evidence);
      
      expect(updates).toHaveLength(3);
      
      // MI should have increased probability due to crushing chest pain
      const miUpdate = updates.find(u => u.diagnosis === 'myocardial_infarction');
      expect(miUpdate).toBeDefined();
      expect(miUpdate!.newProbability).toBeGreaterThan(miUpdate!.oldProbability);
      expect(miUpdate!.likelihoodRatio).toBeGreaterThan(1);
    });

    test('should update probabilities with test results', () => {
      const evidence: DiagnosticEvidence = {
        type: 'test_result',
        name: 'troponin_elevated',
        value: true,
        confidence: 0.95,
        timestamp: new Date()
      };

      const updates = engine.updateWithEvidence(evidence);
      
      const miDiagnosis = engine.getDiagnosis('myocardial_infarction');
      expect(miDiagnosis!.posteriorProbability).toBeGreaterThan(0.02);
      
      // Troponin elevation should strongly suggest MI
      const miUpdate = updates.find(u => u.diagnosis === 'myocardial_infarction');
      expect(miUpdate!.likelihoodRatio).toBeGreaterThan(5); // Should be high for troponin
    });

    test('should handle negative evidence', () => {
      const evidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'fever',
        value: false,
        confidence: 1.0,
        timestamp: new Date()
      };

      const updates = engine.updateWithEvidence(evidence);
      
      // Check that the likelihood ratio for pneumonia with no fever is < 1
      const pneumoniaUpdate = updates.find(u => u.diagnosis === 'pneumonia');
      expect(pneumoniaUpdate).toBeDefined();
      expect(pneumoniaUpdate!.likelihoodRatio).toBeLessThan(1);
      
      // And the raw probability change should be downward
      expect(pneumoniaUpdate!.newProbability).toBeLessThan(pneumoniaUpdate!.oldProbability);
    });

    test('should accumulate evidence over time', () => {
      const evidence1: DiagnosticEvidence = {
        type: 'symptom',
        name: 'chest_pain_crushing',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      const evidence2: DiagnosticEvidence = {
        type: 'symptom',
        name: 'diaphoresis',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      engine.updateWithEvidence(evidence1);
      const probabilityAfterFirst = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      
      engine.updateWithEvidence(evidence2);
      const probabilityAfterSecond = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      
      // Second piece of evidence should further increase probability
      expect(probabilityAfterSecond).toBeGreaterThan(probabilityAfterFirst);
      
      // Should have both pieces of evidence recorded
      const miDiagnosis = engine.getDiagnosis('myocardial_infarction')!;
      expect(miDiagnosis.evidence).toHaveLength(2);
    });

    test('should handle confidence levels', () => {
      const highConfidenceEvidence: DiagnosticEvidence = {
        type: 'test_result',
        name: 'troponin_elevated',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      const lowConfidenceEvidence: DiagnosticEvidence = {
        type: 'test_result',
        name: 'troponin_elevated',
        value: true,
        confidence: 0.3,
        timestamp: new Date()
      };

      // Test high confidence
      engine.updateWithEvidence(highConfidenceEvidence);
      const highConfidenceProb = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      
      // Reset and test low confidence
      engine.reset();
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.updateWithEvidence(lowConfidenceEvidence);
      const lowConfidenceProb = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      
      // High confidence should result in higher probability change
      expect(highConfidenceProb).toBeGreaterThan(lowConfidenceProb);
    });
  });

  describe('Probability Calculations', () => {
    beforeEach(() => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
    });

    test('should maintain probability bounds', () => {
      // Apply multiple strong evidence pieces
      const strongEvidence = [
        { type: 'test_result' as const, name: 'troponin_elevated', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'chest_pain_crushing', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'diaphoresis', value: true, confidence: 1.0 },
        { type: 'demographic' as const, name: 'age_over_65', value: true, confidence: 1.0 }
      ];

      strongEvidence.forEach(evidence => {
        engine.updateWithEvidence({
          ...evidence,
          timestamp: new Date()
        });
      });

      const diagnoses = engine.getRankedDiagnoses();
      diagnoses.forEach(diagnosis => {
        expect(diagnosis.posteriorProbability).toBeGreaterThanOrEqual(0);
        expect(diagnosis.posteriorProbability).toBeLessThanOrEqual(1);
      });
    });

    test('should normalize probabilities when they exceed reasonable bounds', () => {
      // Apply very strong evidence to multiple diagnoses
      const evidence1: DiagnosticEvidence = {
        type: 'test_result',
        name: 'troponin_elevated',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      const evidence2: DiagnosticEvidence = {
        type: 'test_result',
        name: 'chest_xray_infiltrate',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };

      engine.updateWithEvidence(evidence1);
      engine.updateWithEvidence(evidence2);

      const diagnoses = engine.getRankedDiagnoses();
      const totalProbability = diagnoses.reduce((sum, d) => sum + d.posteriorProbability, 0);
      
      // Total probability should be reasonable (not much over 1.0)
      expect(totalProbability).toBeLessThanOrEqual(1.5);
    });
  });

  describe('Information Gain Calculations', () => {
    beforeEach(() => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      engine.initializeDiagnosis('gastroenteritis', 0.15);
    });

    test('should calculate information gain for potential evidence', () => {
      const potentialEvidence = {
        type: 'test_result' as const,
        name: 'troponin_elevated',
        value: true,
        confidence: 1.0
      };

      const informationGain = engine.calculateInformationGain(potentialEvidence);
      
      expect(informationGain).toBeGreaterThanOrEqual(0);
      expect(typeof informationGain).toBe('number');
    });

    test('should show higher information gain for more discriminating evidence', () => {
      const highDiscriminatingEvidence = {
        type: 'test_result' as const,
        name: 'troponin_elevated',
        value: true,
        confidence: 1.0
      };

      const lowDiscriminatingEvidence = {
        type: 'symptom' as const,
        name: 'fatigue',
        value: true,
        confidence: 1.0
      };

      const highGain = engine.calculateInformationGain(highDiscriminatingEvidence);
      const lowGain = engine.calculateInformationGain(lowDiscriminatingEvidence);
      
      expect(highGain).toBeGreaterThan(lowGain);
    });
  });

  describe('Medical Case Scenarios', () => {
    test('acute chest pain workup should favor MI with cardiac evidence', () => {
      // Initialize common chest pain diagnoses
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.03);
      engine.initializeDiagnosis('gastroenteritis', 0.05);

      // Apply chest pain workup evidence
      const evidence = [
        { type: 'symptom' as const, name: 'chest_pain_crushing', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'diaphoresis', value: true, confidence: 1.0 },
        { type: 'test_result' as const, name: 'troponin_elevated', value: true, confidence: 0.95 },
        { type: 'demographic' as const, name: 'age_over_65', value: true, confidence: 1.0 }
      ];

      evidence.forEach(e => {
        engine.updateWithEvidence({
          ...e,
          timestamp: new Date()
        });
      });

      const rankedDiagnoses = engine.getRankedDiagnoses();
      
      // MI should be most likely
      expect(rankedDiagnoses[0].condition).toBe('myocardial_infarction');
      expect(rankedDiagnoses[0].posteriorProbability).toBeGreaterThan(0.1);
    });

    test('respiratory symptoms should favor pneumonia', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      engine.initializeDiagnosis('gastroenteritis', 0.15);

      const evidence = [
        { type: 'symptom' as const, name: 'fever', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'cough', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'shortness_of_breath', value: true, confidence: 1.0 },
        { type: 'test_result' as const, name: 'chest_xray_infiltrate', value: true, confidence: 0.9 }
      ];

      evidence.forEach(e => {
        engine.updateWithEvidence({
          ...e,
          timestamp: new Date()
        });
      });

      const rankedDiagnoses = engine.getRankedDiagnoses();
      
      // Pneumonia should be most likely
      expect(rankedDiagnoses[0].condition).toBe('pneumonia');
    });

    test('gastrointestinal symptoms should favor gastroenteritis', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      engine.initializeDiagnosis('gastroenteritis', 0.15);

      const evidence = [
        { type: 'symptom' as const, name: 'nausea', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'vomiting', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'diarrhea', value: true, confidence: 1.0 },
        { type: 'symptom' as const, name: 'abdominal_pain', value: true, confidence: 1.0 }
      ];

      evidence.forEach(e => {
        engine.updateWithEvidence({
          ...e,
          timestamp: new Date()
        });
      });

      const rankedDiagnoses = engine.getRankedDiagnoses();
      
      // Gastroenteritis should be most likely
      expect(rankedDiagnoses[0].condition).toBe('gastroenteritis');
    });
  });

  describe('Performance Tests', () => {
    test('should handle large numbers of diagnoses efficiently', () => {
      const startTime = Date.now();
      
      // Initialize many diagnoses
      for (let i = 0; i < 100; i++) {
        engine.initializeDiagnosis(`condition_${i}`, 0.01);
      }
      
      // Apply evidence
      const evidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'test_symptom',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };
      
      engine.updateWithEvidence(evidence);
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Should complete within reasonable time (adjust threshold as needed)
      expect(executionTime).toBeLessThan(1000); // 1 second
    });

    test('should handle rapid evidence updates efficiently', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      
      const startTime = Date.now();
      
      // Apply many pieces of evidence rapidly
      for (let i = 0; i < 50; i++) {
        const evidence: DiagnosticEvidence = {
          type: 'symptom',
          name: `symptom_${i}`,
          value: Math.random() > 0.5,
          confidence: Math.random(),
          timestamp: new Date()
        };
        
        engine.updateWithEvidence(evidence);
      }
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      expect(executionTime).toBeLessThan(500); // 0.5 seconds
    });
  });

  describe('Edge Cases', () => {
    test('should handle unknown evidence gracefully', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      
      const unknownEvidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'completely_unknown_symptom',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };
      
      expect(() => {
        engine.updateWithEvidence(unknownEvidence);
      }).not.toThrow();
      
      // Probability should remain unchanged for unknown evidence
      const diagnosis = engine.getDiagnosis('myocardial_infarction');
      expect(diagnosis!.posteriorProbability).toBe(0.02);
    });

    test('should handle zero confidence evidence', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      
      const zeroConfidenceEvidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'chest_pain_crushing',
        value: true,
        confidence: 0.0,
        timestamp: new Date()
      };
      
      const beforeProbability = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      engine.updateWithEvidence(zeroConfidenceEvidence);
      const afterProbability = engine.getDiagnosis('myocardial_infarction')!.posteriorProbability;
      
      // Zero confidence should result in minimal change
      expect(Math.abs(afterProbability - beforeProbability)).toBeLessThan(0.001);
    });

    test('should reset properly', () => {
      engine.initializeDiagnosis('myocardial_infarction', 0.02);
      engine.initializeDiagnosis('pneumonia', 0.05);
      
      const evidence: DiagnosticEvidence = {
        type: 'symptom',
        name: 'chest_pain_crushing',
        value: true,
        confidence: 1.0,
        timestamp: new Date()
      };
      
      engine.updateWithEvidence(evidence);
      expect(engine.getRankedDiagnoses()).toHaveLength(2);
      
      engine.reset();
      expect(engine.getRankedDiagnoses()).toHaveLength(0);
    });
  });
});