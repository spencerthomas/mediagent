/**
 * Bayesian Engine Null Safety Tests
 * Tests handling of null/undefined values in medical case data
 */

import { PhysicianAgent, PHYSICIAN_AGENTS } from '../physician_agents';
import { MedicalDiagnosticStateType } from '../medical_state';
import { HumanMessage } from '@langchain/core/messages';

// Mock chat model for testing
const mockLoadChatModel = async (modelName: string) => ({
  invoke: async (_messages: any[]) => ({
    content: `Mock response for ${modelName}: Analysis complete.`
  })
});

describe('Bayesian Engine Null Safety', () => {
  let hypothesisAgent: PhysicianAgent;

  beforeEach(() => {
    hypothesisAgent = new PhysicianAgent(
      PHYSICIAN_AGENTS.hypothesis,
      mockLoadChatModel
    );
  });

  test('should handle null chief complaint gracefully', async () => {
    const testState: MedicalDiagnosticStateType = {
      messages: [new HumanMessage("Test case with null chief complaint")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-null-chief",
        chiefComplaint: null as any, // Simulate null value
        demographics: { age: 65, gender: 'male' },
        historyOfPresentIllness: "Patient presents with symptoms",
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [],
      cumulativeCost: 0,
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

    const config = { configurable: { model: 'gpt-4' } };
    const context = "Analyze this case with null chief complaint.";

    // Should not throw error
    expect(async () => {
      await hypothesisAgent.generateResponse(testState, context, config);
    }).not.toThrow();

    // Test Bayesian analysis directly
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);
    expect(bayesianAnalysis).toBeDefined();
    expect(typeof bayesianAnalysis).toBe('string');
  });

  test('should handle null history of present illness gracefully', async () => {
    const testState: MedicalDiagnosticStateType = {
      messages: [new HumanMessage("Test case with null history")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-null-history",
        chiefComplaint: "chest pain",
        demographics: { age: 65, gender: 'male' },
        historyOfPresentIllness: null as any, // Simulate null value
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [],
      cumulativeCost: 0,
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

    // Test Bayesian analysis directly - should not throw
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);
    expect(bayesianAnalysis).toBeDefined();
    expect(typeof bayesianAnalysis).toBe('string');
  });

  test('should handle null demographics gracefully', async () => {
    const testState: MedicalDiagnosticStateType = {
      messages: [new HumanMessage("Test case with null demographics")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-null-demographics",
        chiefComplaint: "chest pain",
        demographics: null as any, // Simulate null value
        historyOfPresentIllness: "Patient presents with chest pain",
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [],
      cumulativeCost: 0,
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

    // Test Bayesian analysis directly - should not throw
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);
    expect(bayesianAnalysis).toBeDefined();
    expect(typeof bayesianAnalysis).toBe('string');
  });

  test('should handle missing age in demographics gracefully', async () => {
    const testState: MedicalDiagnosticStateType = {
      messages: [new HumanMessage("Test case with missing age")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-missing-age",
        chiefComplaint: "chest pain",
        demographics: { age: null as any, gender: 'male' }, // Simulate null age
        historyOfPresentIllness: "Patient presents with chest pain",
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [],
      cumulativeCost: 0,
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

    // Test Bayesian analysis directly - should not throw
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);
    expect(bayesianAnalysis).toBeDefined();
    expect(typeof bayesianAnalysis).toBe('string');
    
    // Should not include age-based evidence
    expect(bayesianAnalysis).not.toContain('age_over_65');
  });

  test('should handle completely empty case info gracefully', async () => {
    const testState: MedicalDiagnosticStateType = {
      messages: [new HumanMessage("Test case with minimal info")],
      differentialDiagnoses: [],
      availableCaseInfo: {
        patientId: "test-patient-minimal",
        chiefComplaint: "",
        demographics: { age: 0, gender: 'unknown' },
        historyOfPresentIllness: "",
        pastMedicalHistory: [],
        medications: [],
        allergies: [],
        familyHistory: [],
        socialHistory: ""
      },
      revealedInformation: [],
      diagnosticTests: [],
      cumulativeCost: 0,
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

    // Test Bayesian analysis directly - should not throw and should initialize default diagnoses
    const bayesianAnalysis = (hypothesisAgent as any).prepareBayesianAnalysis(testState);
    expect(bayesianAnalysis).toBeDefined();
    expect(typeof bayesianAnalysis).toBe('string');
    
    // Should contain the default diagnoses when no differential diagnoses exist
    expect(bayesianAnalysis).toContain('myocardial infarction');
    expect(bayesianAnalysis).toContain('pneumonia');
    expect(bayesianAnalysis).toContain('gastroenteritis');
  });
});