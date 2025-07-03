/**
 * Medical Agent Configuration
 * Defines configurable parameters for the medical diagnostic agent
 */

import { Annotation } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";

export const MEDICAL_SYSTEM_PROMPT = `You are a medical diagnostic superintelligence system composed of 5 specialized physician agents working collaboratively to diagnose complex medical cases.

Your mission is to provide accurate, cost-effective, and comprehensive medical diagnostic analysis through structured deliberation and evidence-based reasoning.

Core principles:
- Systematic diagnostic approach
- Evidence-based reasoning  
- Cost-conscious decision making
- Bias detection and mitigation
- Collaborative deliberation
- Patient safety first

You will process medical cases through a Chain of Debate workflow involving specialized physician agents:
- Dr. Hypothesis: Differential diagnosis specialist
- Dr. Test-Chooser: Diagnostic test selection expert
- Dr. Challenger: Bias detection and alternative reasoning
- Dr. Stewardship: Cost-effective care advocate
- Dr. Checklist: Quality assurance specialist

System time: {system_time}`;

export const MedicalConfigurationSchema = Annotation.Root({
  /**
   * The system prompt template for the medical diagnostic agent
   */
  systemPromptTemplate: Annotation<string>,

  /**
   * The name of the language model to be used by the medical agents
   */
  model: Annotation<string>,

  /**
   * Budget limit for diagnostic workup (in USD)
   */
  costBudget: Annotation<number>,

  /**
   * Maximum number of debate rounds before forced conclusion
   */
  maxDebateRounds: Annotation<number>,

  /**
   * Confidence threshold for diagnostic decisions (0-1)
   */
  confidenceThreshold: Annotation<number>,

  /**
   * Participating physician agent roles
   */
  participatingAgents: Annotation<string[]>,

  /**
   * Enable/disable gatekeeper information control
   */
  enableGatekeeper: Annotation<boolean>,

  /**
   * Enable/disable bias detection
   */
  enableBiasDetection: Annotation<boolean>,

  /**
   * Enable/disable cost tracking
   */
  enableCostTracking: Annotation<boolean>
});

export function ensureMedicalConfiguration(
  config: RunnableConfig,
): typeof MedicalConfigurationSchema.State {
  const configurable = config.configurable ?? {};
  
  return {
    systemPromptTemplate:
      configurable.systemPromptTemplate ?? MEDICAL_SYSTEM_PROMPT,
    model: configurable.model ?? "o3-mini",
    costBudget: configurable.costBudget ?? 1000,
    maxDebateRounds: configurable.maxDebateRounds ?? 5,
    confidenceThreshold: configurable.confidenceThreshold ?? 0.8,
    participatingAgents: configurable.participatingAgents ?? [
      'hypothesis', 
      'test_chooser', 
      'challenger', 
      'stewardship', 
      'checklist'
    ],
    enableGatekeeper: configurable.enableGatekeeper ?? true,
    enableBiasDetection: configurable.enableBiasDetection ?? true,
    enableCostTracking: configurable.enableCostTracking ?? true
  };
}

export type MedicalConfigurationType = typeof MedicalConfigurationSchema.State;