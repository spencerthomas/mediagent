/**
 * Bayesian Engine Integration Tests
 * Tests the integration of Bayesian reasoning with physician agents
 */

import { PhysicianAgent, PHYSICIAN_AGENTS } from '../physician_agents.js';
import { MedicalDiagnosticStateType } from '../medical_state.js';
import { HumanMessage } from '@langchain/core/messages';

// Mock chat model for testing
const mockLoadChatModel = async (modelName: string) => ({
  invoke: async (_messages: any[]) => ({
    content: `Mock response for ${modelName}: Bayesian analysis suggests myocardial infarction has highest probability based on crushing chest pain and elevated troponin.`
  })
});

describe('Bayesian Engine Integration', () => {
  let hypothesisAgent: PhysicianAgent;
  let testState: MedicalDiagnosticStateType;

  beforeEach(() => {
    hypothesisAgent = new PhysicianAgent(
      PHYSICIAN_AGENTS.hypothesis,
      mockLoadChatModel
    );

    testState = {
      messages: [new HumanMessage("65-year-old male presents with crushing chest pain and diaphoresis. Troponin elevated.")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-001",
        chiefComplaint: "crushing chest pain",
        demographics: { age: 65, gender: 'male' },
        historyOfPresentIllness: "Sudden onset crushing chest pain with diaphoresis 2 hours ago",
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [
        {
          testName: "Troponin I",
          testType: "blood",
          result: "elevated (0.8 ng/mL)",
          cost: 75,
          diagnosticValue: 0.95,
          requestedBy: "emergency_physician",
          timestamp: new Date()
        }
      ],
      cumulativeCost: 75,
      costBudget: 1000,
      currentPhase: 'deliberation',
      debateRound: 1,
      agentTurns: [],
      confidenceLevel: 0.6,
      readyForDiagnosis: false,
      pendingQuestions: [],
      awaitingUserInput: false,
      questionHistory: [],
      userResponses: [],
      interactionRound: 0,
      requiredInformationGathered: [],
      biasesDetected: [],
      reasoningQuality: 0.8,
      finalDiagnosis: undefined
    };
  });

  test('should integrate Bayesian analysis with hypothesis agent', async () => {
    const config = { configurable: { model: 'gpt-4' } };
    const context = "Please provide differential diagnosis with Bayesian probability updating.";

    const agentTurn = await hypothesisAgent.generateResponse(testState, context, config);

    expect(agentTurn).toBeDefined();
    expect(agentTurn.agentRole).toBe('hypothesis');
    expect(agentTurn.reasoning).toContain('Bayesian');
    expect(agentTurn.reasoning).toContain('myocardial infarction');
  });

  test('should apply Bayesian reasoning to medical case with multiple evidence', async () => {
    // Add multiple pieces of evidence
    testState.availableCaseInfo.chiefComplaint = "crushing chest pain with shortness of breath";
    testState.availableCaseInfo.historyOfPresentIllness = "Patient reports severe crushing chest pain radiating to left arm, accompanied by nausea and diaphoresis. Started 2 hours ago while at rest.";
    
    testState.diagnosticTests.push({
      testName: "ECG",
      testType: "imaging",
      result: "ST elevation in leads II, III, aVF",
      cost: 50,
      diagnosticValue: 0.85,
      requestedBy: "emergency_physician",
      timestamp: new Date()
    });

    // Test the Bayesian analysis preparation directly
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);

    expect(bayesianAnalysis).toBeDefined();
    expect(bayesianAnalysis).toContain('FORMAL BAYESIAN PROBABILITY ANALYSIS');
    expect(bayesianAnalysis).toContain('Updated Diagnosis Probabilities');
    expect(bayesianAnalysis).toContain('Evidence Applied');
    expect(bayesianAnalysis).toContain('Information Gain Analysis');
    expect(bayesianAnalysis).toContain('myocardial infarction');
    expect(bayesianAnalysis).toContain('chest pain crushing');
    expect(bayesianAnalysis).toContain('troponin elevated');
  });

  test('should handle case with existing differential diagnoses', async () => {
    // Pre-populate with existing diagnoses
    testState.differentialDiagnoses = [
      {
        condition: 'myocardial infarction',
        probability: 70,
        supportingEvidence: ['chest pain', 'elevated troponin'],
        reasoning: 'Classic presentation with biomarker elevation'
      },
      {
        condition: 'pneumonia',
        probability: 15,
        supportingEvidence: ['age'],
        reasoning: 'Less likely given presentation'
      }
    ];

    // Test the Bayesian analysis preparation with existing diagnoses
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);

    expect(bayesianAnalysis).toBeDefined();
    expect(bayesianAnalysis).toContain('FORMAL BAYESIAN PROBABILITY ANALYSIS');
    expect(bayesianAnalysis).toContain('myocardial infarction');
    expect(bayesianAnalysis).toContain('troponin elevated');
    
    // Should have updated probabilities based on existing diagnoses
    expect(bayesianAnalysis).toContain('Updated Diagnosis Probabilities');
  });

  test('should provide information gain analysis for test selection', async () => {
    // Test the Bayesian analysis directly for information gain calculations
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);

    expect(bayesianAnalysis).toBeDefined();
    expect(bayesianAnalysis).toContain('Information Gain Analysis');
    expect(bayesianAnalysis).toContain('bits');
    
    // Should include potential tests
    expect(bayesianAnalysis).toContain('troponin elevated');
    expect(bayesianAnalysis).toContain('chest xray infiltrate');
    expect(bayesianAnalysis).toContain('diaphoresis');
  });
});