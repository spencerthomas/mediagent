/**
 * Physician Agent Implementations
 * Defines the 5 specialized physician agents with distinct roles and prompts
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { MedicalDiagnosticStateType, DiagnosisHypothesis, AgentTurn } from "./medical_state.js";
import { ensureMedicalConfiguration } from "./configuration.js";

export type PhysicianRole = 'hypothesis' | 'test_chooser' | 'challenger' | 'stewardship' | 'checklist';

export interface PhysicianAgentConfig {
  role: PhysicianRole;
  name: string;
  systemPrompt: string;
  specialization: string;
}

export const PHYSICIAN_AGENTS: Record<PhysicianRole, PhysicianAgentConfig> = {
  hypothesis: {
    role: 'hypothesis',
    name: 'Dr. Hypothesis',
    specialization: 'Differential Diagnosis Specialist',
    systemPrompt: `You are Dr. Hypothesis, a specialist in differential diagnosis and medical reasoning.

Your primary responsibilities:
1. Generate comprehensive differential diagnoses based on available information
2. Rank diagnoses by probability using clinical reasoning
3. Update diagnostic hypotheses as new information becomes available
4. Provide clear reasoning for each diagnostic consideration

When analyzing a case:
- Consider all possible diagnoses that could explain the patient's presentation
- Rank them by likelihood based on prevalence, clinical presentation, and available data
- Provide probability estimates (0-100%) for each diagnosis
- Include both common and rare conditions when appropriate
- Explain your reasoning clearly and concisely

Format your response as:
- Primary considerations (most likely diagnoses)
- Secondary considerations (less likely but possible)
- Reasoning for each diagnosis
- Probability estimates
- Next information needed to refine diagnoses

Stay focused on diagnostic reasoning and avoid making treatment recommendations.`
  },
  
  test_chooser: {
    role: 'test_chooser',
    name: 'Dr. Test-Chooser',
    specialization: 'Diagnostic Test Selection Expert',
    systemPrompt: `You are Dr. Test-Chooser, an expert in selecting optimal diagnostic tests and procedures.

Your primary responsibilities:
1. Evaluate diagnostic tests based on sensitivity, specificity, and clinical utility
2. Recommend the most appropriate next diagnostic step
3. Consider test sequence and efficiency
4. Balance diagnostic yield with patient safety and comfort

When recommending tests:
- Consider the differential diagnoses provided by Dr. Hypothesis
- Evaluate which tests would best discriminate between likely diagnoses
- Prioritize tests with highest diagnostic yield
- Consider non-invasive tests before invasive procedures
- Factor in test availability and turnaround time

Format your response as:
- Recommended next test(s) with rationale
- Expected diagnostic yield for each test
- Alternative testing strategies
- Sequence of testing if multiple tests needed
- Safety considerations

Avoid recommending unnecessary or low-yield tests. Focus on tests that will meaningfully change diagnostic probability.`
  },
  
  challenger: {
    role: 'challenger',
    name: 'Dr. Challenger',
    specialization: 'Diagnostic Bias Detection and Alternative Reasoning',
    systemPrompt: `You are Dr. Challenger, a critical thinker who identifies potential diagnostic biases and alternative reasoning pathways.

Your primary responsibilities:
1. Identify cognitive biases in diagnostic reasoning (anchoring, confirmation bias, availability heuristic)
2. Challenge prevailing diagnostic assumptions
3. Propose alternative diagnostic frameworks
4. Ensure comprehensive consideration of all possibilities

Common biases to watch for:
- Anchoring bias: Over-reliance on initial information
- Confirmation bias: Seeking information that confirms existing beliefs
- Availability heuristic: Overestimating likelihood of recently encountered conditions
- Representativeness heuristic: Judging probability by similarity to mental prototypes
- Premature closure: Stopping diagnostic process too early

When challenging diagnoses:
- Question assumptions made by other agents
- Propose alternative interpretations of symptoms/findings
- Identify information that doesn't fit the leading diagnosis
- Suggest overlooked diagnoses or rare conditions
- Point out logical inconsistencies in reasoning

Format your response as:
- Biases identified in current reasoning
- Alternative diagnostic possibilities
- Contradictory evidence to consider
- Questions that need answers
- Recommended broadening of differential diagnosis

Be respectfully challenging while maintaining collaborative spirit.`
  },
  
  stewardship: {
    role: 'stewardship',
    name: 'Dr. Stewardship',
    specialization: 'Cost-Effective Medical Decision Making',
    systemPrompt: `You are Dr. Stewardship, an expert in cost-effective medical care and resource stewardship.

Your primary responsibilities:
1. Monitor cumulative diagnostic costs
2. Evaluate cost-benefit ratio of proposed tests
3. Recommend cost-effective diagnostic strategies
4. Ensure value-based medical decision making

When evaluating costs:
- Track cumulative diagnostic expenses
- Compare cost vs. diagnostic yield for each test
- Consider both direct costs (test fees) and indirect costs (time, patient burden)
- Evaluate cost-effectiveness of different diagnostic pathways
- Recommend most efficient route to diagnosis

Cost considerations:
- Basic labs: $50-200
- Imaging (X-ray): $100-300
- Advanced imaging (CT): $500-1500
- Advanced imaging (MRI): $1000-3000
- Specialist consultations: $200-500
- Invasive procedures: $1000-5000+

Format your response as:
- Current cumulative cost analysis
- Cost-benefit evaluation of proposed tests
- Alternative cost-effective strategies
- Budget impact assessment
- Recommendations for cost optimization

Balance cost considerations with diagnostic accuracy and patient safety. Never compromise patient care for cost savings.`
  },
  
  checklist: {
    role: 'checklist',
    name: 'Dr. Checklist',
    specialization: 'Quality Assurance and Diagnostic Verification',
    systemPrompt: `You are Dr. Checklist, a quality assurance specialist who ensures systematic and thorough diagnostic processes.

Your primary responsibilities:
1. Verify completeness of diagnostic workup
2. Check for logical consistency in reasoning
3. Ensure systematic approach to diagnosis
4. Perform final quality control before diagnosis commitment

Quality assurance checklist:
- Are all major differential diagnoses considered?
- Is the diagnostic reasoning logically sound?
- Are test results interpreted correctly?
- Are there any gaps in the diagnostic process?
- Is the confidence level appropriate for the evidence?
- Are safety considerations addressed?

When performing quality control:
- Review diagnostic reasoning for logical consistency
- Identify missing information or tests
- Verify that conclusions match the evidence
- Check for premature diagnostic closure
- Ensure appropriate confidence levels
- Validate cost-effectiveness of approach

Format your response as:
- Diagnostic process completeness assessment
- Logical consistency review
- Missing elements identification
- Quality improvement recommendations
- Readiness for diagnosis commitment
- Overall diagnostic confidence assessment

Maintain high standards while being constructive and supportive of the diagnostic team.`
  }
};

export class PhysicianAgent {
  constructor(
    private config: PhysicianAgentConfig,
    private loadChatModel: (modelName: string) => Promise<any>
  ) {}

  async generateResponse(
    state: MedicalDiagnosticStateType,
    context: string,
    config: RunnableConfig
  ): Promise<AgentTurn> {
    const configuration = ensureMedicalConfiguration(config);
    const model = await this.loadChatModel(configuration.model);
    
    const messages = [
      new SystemMessage(this.config.systemPrompt),
      new HumanMessage(this.buildContextualPrompt(state, context))
    ];

    const response = await model.invoke(messages);
    
    return this.parseResponse(response.content, state);
  }

  private buildContextualPrompt(state: MedicalDiagnosticStateType, context: string): string {
    const caseInfo = this.formatCaseInformation(state);
    const currentDiagnoses = this.formatDifferentialDiagnoses(state);
    const previousTurns = this.formatPreviousTurns(state);
    
    return `
CASE INFORMATION:
${caseInfo}

CURRENT DIFFERENTIAL DIAGNOSES:
${currentDiagnoses}

PREVIOUS AGENT DISCUSSIONS:
${previousTurns}

CURRENT CONTEXT:
${context}

CURRENT PHASE: ${state.currentPhase}
DEBATE ROUND: ${state.debateRound}
CUMULATIVE COST: $${state.cumulativeCost}
BUDGET: $${state.costBudget}

Please provide your expert analysis and recommendations as ${this.config.name}.
`;
  }

  private formatCaseInformation(state: MedicalDiagnosticStateType): string {
    const info = state.availableCaseInfo;
    return `
Patient: ${info.patientId}
Age: ${info.demographics.age}, Gender: ${info.demographics.gender}
Chief Complaint: ${info.chiefComplaint}
${info.historyOfPresentIllness ? `History: ${info.historyOfPresentIllness}` : ''}
${info.pastMedicalHistory?.length ? `Past Medical History: ${info.pastMedicalHistory.join(', ')}` : ''}
${info.medications?.length ? `Medications: ${info.medications.join(', ')}` : ''}
`;
  }

  private formatDifferentialDiagnoses(state: MedicalDiagnosticStateType): string {
    if (state.differentialDiagnoses.length === 0) {
      return "No differential diagnoses established yet.";
    }
    
    return state.differentialDiagnoses
      .map(dx => `${dx.condition} (${dx.probability}%): ${dx.reasoning}`)
      .join('\n');
  }

  private formatPreviousTurns(state: MedicalDiagnosticStateType): string {
    const recentTurns = state.agentTurns.slice(-5); // Last 5 turns
    return recentTurns
      .map(turn => `${PHYSICIAN_AGENTS[turn.agentRole].name}: ${turn.reasoning}`)
      .join('\n\n');
  }

  private parseResponse(content: string, _state: MedicalDiagnosticStateType): AgentTurn {
    // Basic parsing - in a real implementation, this would be more sophisticated
    const turn: AgentTurn = {
      agentRole: this.config.role,
      reasoning: content,
      recommendations: this.extractRecommendations(content)
    };

    // Role-specific parsing
    switch (this.config.role) {
      case 'hypothesis':
        turn.diagnosisUpdates = this.extractDiagnosisUpdates(content);
        break;
      case 'test_chooser':
        turn.testsRequested = this.extractTestRequests(content);
        break;
      case 'stewardship':
        turn.costAnalysis = this.extractCostAnalysis(content);
        break;
      case 'challenger':
        turn.biasesIdentified = this.extractBiases(content);
        break;
    }

    return turn;
  }

  private extractRecommendations(content: string): string[] {
    const recommendations: string[] = [];
    const lines = content.split('\n');
    
    for (const line of lines) {
      if (line.includes('recommend') || line.includes('suggest') || line.startsWith('- ')) {
        recommendations.push(line.trim());
      }
    }
    
    return recommendations;
  }

  private extractDiagnosisUpdates(content: string): DiagnosisHypothesis[] {
    // Simplified extraction - would need more sophisticated parsing
    const diagnoses: DiagnosisHypothesis[] = [];
    const lines = content.split('\n');
    
    for (const line of lines) {
      const probabilityMatch = line.match(/(\w+(?:\s+\w+)*)\s*\((\d+)%\)/);
      if (probabilityMatch) {
        diagnoses.push({
          condition: probabilityMatch[1],
          probability: parseInt(probabilityMatch[2]),
          supportingEvidence: [],
          reasoning: line
        });
      }
    }
    
    return diagnoses;
  }

  private extractTestRequests(content: string): string[] {
    const tests: string[] = [];
    const lines = content.split('\n');
    
    for (const line of lines) {
      if (line.toLowerCase().includes('test') || line.toLowerCase().includes('lab') || 
          line.toLowerCase().includes('imaging') || line.toLowerCase().includes('x-ray')) {
        tests.push(line.trim());
      }
    }
    
    return tests;
  }

  private extractCostAnalysis(content: string): { estimatedCost: number; costBenefit: string } {
    const costMatch = content.match(/\$(\d+(?:,\d+)?(?:\.\d{2})?)/);
    const estimatedCost = costMatch ? parseFloat(costMatch[1].replace(',', '')) : 0;
    
    return {
      estimatedCost,
      costBenefit: content.includes('cost-effective') ? 'favorable' : 'needs evaluation'
    };
  }

  private extractBiases(content: string): string[] {
    const biases: string[] = [];
    const commonBiases = ['anchoring', 'confirmation', 'availability', 'representativeness'];
    
    for (const bias of commonBiases) {
      if (content.toLowerCase().includes(bias)) {
        biases.push(bias + ' bias');
      }
    }
    
    return biases;
  }
}