/**
 * Tests for the Medical Diagnostic Agent
 */

import { medicalGraph } from "../medical_graph.js";
import { MedicalDiagnosticAnnotation } from "../medical_state.js";
import { HumanMessage } from "@langchain/core/messages";

describe("Medical Diagnostic Agent", () => {
  test("should initialize and process a basic medical case", async () => {
    const initialState = {
      messages: [new HumanMessage("55-year-old male presents with chest pain and shortness of breath for 2 hours. Pain is crushing, radiates to left arm. Patient appears diaphoretic and anxious.")]
    };

    const result = await medicalGraph.invoke(initialState, {
      configurable: {
        model: "o3-mini",
        systemPromptTemplate: "You are a medical diagnostic AI assistant. Analyze the provided medical case and provide diagnostic insights."
      }
    });

    // Verify the agent processed the case
    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(1);
    
    // Verify case information was extracted
    expect(result.availableCaseInfo).toBeDefined();
    expect(result.availableCaseInfo.chiefComplaint).toBeDefined();
    
    // Verify workflow progressed through phases
    expect(result.currentPhase).toBeDefined();
    
    // Verify diagnostic state was initialized
    expect(result.differentialDiagnoses).toBeDefined();
    expect(Array.isArray(result.differentialDiagnoses)).toBe(true);
  });

  test("should handle state management correctly", () => {
    const testState = MedicalDiagnosticAnnotation.State;
    
    // Verify state structure
    expect(testState.messages).toEqual([]);
    expect(testState.differentialDiagnoses).toEqual([]);
    expect(testState.cumulativeCost).toBe(0);
    expect(testState.costBudget).toBe(1000);
    expect(testState.currentPhase).toBe('case_presentation');
    expect(testState.debateRound).toBe(0);
    expect(testState.confidenceLevel).toBe(0);
    expect(testState.readyForDiagnosis).toBe(false);
  });
});