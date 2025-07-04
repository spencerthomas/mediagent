/**
 * Chain of Debate Implementation
 * Orchestrates structured deliberation between physician agents
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import { MedicalDiagnosticStateType, DiagnosticPhase, AgentTurn, DiagnosisHypothesis } from "./medical_state.js";
import { PhysicianAgent, PhysicianRole, PHYSICIAN_AGENTS } from "./physician_agents.js";

export interface DebateConfiguration {
  maxRounds: number;
  confidenceThreshold: number;
  costBudgetLimit: number;
  participatingAgents: PhysicianRole[];
}

export class ChainOfDebate {
  private physicianAgents: Map<PhysicianRole, PhysicianAgent> = new Map();
  private debateConfig: DebateConfiguration;

  constructor(
    private loadChatModel: (modelName: string) => Promise<any>,
    config: Partial<DebateConfiguration> = {}
  ) {
    this.debateConfig = {
      maxRounds: 2,
      confidenceThreshold: 0.8,
      costBudgetLimit: 1000,
      participatingAgents: ['hypothesis', 'test_chooser', 'challenger', 'stewardship', 'checklist'],
      ...config
    };

    // Initialize physician agents
    for (const role of this.debateConfig.participatingAgents) {
      this.physicianAgents.set(
        role,
        new PhysicianAgent(PHYSICIAN_AGENTS[role], this.loadChatModel)
      );
    }
  }

  async orchestrateDebate(
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<Partial<MedicalDiagnosticStateType>> {
    const updates: Partial<MedicalDiagnosticStateType> = {};

    // Determine debate sequence based on current phase
    const sequence = this.getDebateSequence(state.currentPhase);
    
    // Execute debate rounds
    for (let round = 0; round < this.debateConfig.maxRounds; round++) {
      const roundUpdates = await this.executeDebateRound(state, sequence, round, config);
      
      // Apply updates to state
      Object.assign(state, roundUpdates);
      Object.assign(updates, roundUpdates);

      // Check for early termination conditions
      if (this.shouldTerminateDebate(state)) {
        break;
      }
    }

    // Final synthesis and decision
    const finalUpdates = await this.synthesizeDebateResults(state, config);
    Object.assign(updates, finalUpdates);

    return updates;
  }

  private async executeDebateRound(
    state: MedicalDiagnosticStateType,
    sequence: PhysicianRole[],
    round: number,
    config: RunnableConfig
  ): Promise<Partial<MedicalDiagnosticStateType>> {
    const agentTurns: AgentTurn[] = [];
    let updatedDiagnoses = [...state.differentialDiagnoses];
    let cumulativeCost = 0;
    const biasesDetected: string[] = [];

    for (const role of sequence) {
      const agent = this.physicianAgents.get(role);
      if (!agent) continue;

      const context = this.buildDebateContext(state, role, round, agentTurns);
      const agentTurn = await agent.generateResponse(state, context, config);
      
      agentTurns.push(agentTurn);

      // Process agent-specific updates
      if (agentTurn.diagnosisUpdates) {
        updatedDiagnoses = this.updateDifferentialDiagnoses(
          updatedDiagnoses,
          agentTurn.diagnosisUpdates
        );
      }

      if (agentTurn.costAnalysis) {
        cumulativeCost += agentTurn.costAnalysis.estimatedCost;
      }

      if (agentTurn.biasesIdentified) {
        biasesDetected.push(...agentTurn.biasesIdentified);
      }
    }

    return {
      agentTurns: [...state.agentTurns, ...agentTurns],
      differentialDiagnoses: updatedDiagnoses,
      cumulativeCost: state.cumulativeCost + cumulativeCost,
      biasesDetected: [...state.biasesDetected, ...biasesDetected],
      debateRound: round + 1,
      messages: [
        ...state.messages,
        new AIMessage(`Debate Round ${round + 1} completed with ${agentTurns.length} agent contributions.`)
      ]
    };
  }

  private buildDebateContext(
    state: MedicalDiagnosticStateType,
    currentRole: PhysicianRole,
    round: number,
    currentTurns: AgentTurn[]
  ): string {
    const context = [`This is round ${round + 1} of the diagnostic debate.`];
    
    // Add phase-specific context
    switch (state.currentPhase) {
      case 'initial_assessment':
        context.push("Focus on establishing initial differential diagnoses based on available information.");
        break;
      case 'information_gathering':
        context.push("Determine what additional information is needed to refine the diagnosis.");
        break;
      case 'test_selection':
        context.push("Select the most appropriate diagnostic tests based on current hypotheses.");
        break;
      case 'deliberation':
        context.push("Deliberate on all available information to reach diagnostic conclusions.");
        break;
      case 'final_diagnosis':
        context.push("Make final diagnostic decisions based on all available evidence.");
        break;
    }

    // Add role-specific context
    switch (currentRole) {
      case 'hypothesis':
        context.push("Provide updated differential diagnoses with probability assessments.");
        break;
      case 'test_chooser':
        context.push("Recommend diagnostic tests that will best discriminate between hypotheses.");
        break;
      case 'challenger':
        context.push("Challenge current assumptions and identify potential biases.");
        break;
      case 'stewardship':
        context.push("Evaluate cost-effectiveness of proposed diagnostic strategies.");
        break;
      case 'checklist':
        context.push("Perform quality assurance on the diagnostic process.");
        break;
    }

    // Add current turn context
    if (currentTurns.length > 0) {
      context.push("\nCurrent debate contributions:");
      for (const turn of currentTurns) {
        context.push(`${PHYSICIAN_AGENTS[turn.agentRole].name}: ${turn.reasoning.substring(0, 200)}...`);
      }
    }

    return context.join('\n');
  }

  private getDebateSequence(phase: DiagnosticPhase): PhysicianRole[] {
    switch (phase) {
      case 'case_presentation':
      case 'initial_assessment':
        return ['hypothesis', 'challenger', 'checklist'];
      
      case 'information_gathering':
        return ['hypothesis', 'test_chooser', 'stewardship', 'challenger'];
      
      case 'test_selection':
        return ['test_chooser', 'stewardship', 'challenger', 'checklist'];
      
      case 'deliberation':
        return ['hypothesis', 'challenger', 'stewardship', 'checklist'];
      
      case 'final_diagnosis':
        return ['hypothesis', 'checklist', 'stewardship'];
      
      default:
        return ['hypothesis', 'test_chooser', 'challenger', 'stewardship', 'checklist'];
    }
  }

  private updateDifferentialDiagnoses(
    current: DiagnosisHypothesis[],
    updates: DiagnosisHypothesis[]
  ): DiagnosisHypothesis[] {
    const diagnoses = [...current];
    
    for (const update of updates) {
      const existingIndex = diagnoses.findIndex(d => d.condition === update.condition);
      
      if (existingIndex >= 0) {
        // Update existing diagnosis
        diagnoses[existingIndex] = {
          ...diagnoses[existingIndex],
          probability: update.probability,
          supportingEvidence: [
            ...diagnoses[existingIndex].supportingEvidence,
            ...update.supportingEvidence
          ],
          reasoning: update.reasoning
        };
      } else {
        // Add new diagnosis
        diagnoses.push(update);
      }
    }

    // Sort by probability (descending)
    return diagnoses.sort((a, b) => b.probability - a.probability);
  }

  private shouldTerminateDebate(state: MedicalDiagnosticStateType): boolean {
    // Check confidence threshold
    if (state.confidenceLevel >= this.debateConfig.confidenceThreshold) {
      return true;
    }

    // Check budget constraints
    if (state.cumulativeCost >= this.debateConfig.costBudgetLimit) {
      return true;
    }

    // Check if ready for diagnosis
    if (state.readyForDiagnosis) {
      return true;
    }

    return false;
  }

  private async synthesizeDebateResults(
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<Partial<MedicalDiagnosticStateType>> {
    const checklistAgent = this.physicianAgents.get('checklist');
    if (!checklistAgent) {
      throw new Error('Checklist agent not available for synthesis');
    }

    // Get final assessment from checklist agent
    const synthesisContext = `
Please provide a final synthesis of the diagnostic debate:
1. Assess diagnostic confidence level (0-1)
2. Determine if ready for final diagnosis
3. Identify the most likely diagnosis
4. Provide overall quality assessment
`;

    await checklistAgent.generateResponse(
      state,
      synthesisContext,
      config
    );

    // Determine final diagnosis
    const finalDiagnosis = state.differentialDiagnoses.length > 0 
      ? state.differentialDiagnoses[0] 
      : undefined;

    // Calculate confidence based on top diagnosis probability and debate quality
    const confidenceLevel = finalDiagnosis 
      ? finalDiagnosis.probability / 100 
      : 0;

    // Determine next phase
    const nextPhase = this.determineNextPhase(state, confidenceLevel);

    return {
      finalDiagnosis,
      confidenceLevel,
      readyForDiagnosis: confidenceLevel >= this.debateConfig.confidenceThreshold,
      currentPhase: nextPhase,
      reasoningQuality: this.assessReasoningQuality(state),
      messages: [
        ...state.messages,
        new AIMessage(`Debate synthesis completed. Confidence: ${(confidenceLevel * 100).toFixed(1)}%`)
      ]
    };
  }

  private determineNextPhase(
    state: MedicalDiagnosticStateType,
    confidenceLevel: number
  ): DiagnosticPhase {
    if (confidenceLevel >= this.debateConfig.confidenceThreshold) {
      return 'final_diagnosis';
    }

    switch (state.currentPhase) {
      case 'case_presentation':
        return 'initial_assessment';
      case 'initial_assessment':
        return 'information_gathering';
      case 'information_gathering':
        return 'test_selection';
      case 'test_selection':
        return 'deliberation';
      case 'deliberation':
        return confidenceLevel > 0.6 ? 'final_diagnosis' : 'information_gathering';
      default:
        return 'final_diagnosis';
    }
  }

  private assessReasoningQuality(state: MedicalDiagnosticStateType): number {
    let qualityScore = 0;
    
    // Factor in number of agent contributions
    const contributionScore = Math.min(state.agentTurns.length / 10, 1);
    qualityScore += contributionScore * 0.3;
    
    // Factor in diagnostic diversity
    const diversityScore = Math.min(state.differentialDiagnoses.length / 5, 1);
    qualityScore += diversityScore * 0.2;
    
    // Factor in bias detection
    const biasScore = Math.min(state.biasesDetected.length / 3, 1);
    qualityScore += biasScore * 0.2;
    
    // Factor in cost consideration
    const costScore = state.cumulativeCost > 0 ? 0.3 : 0;
    qualityScore += costScore;
    
    return Math.min(qualityScore, 1);
  }
}