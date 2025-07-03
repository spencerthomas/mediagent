/**
 * Patient Question Agent
 * Generates targeted follow-up questions based on diagnostic needs
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { HumanMessage } from "@langchain/core/messages";
import { 
  MedicalDiagnosticStateType, 
  PatientQuestion
} from "./medical_state.js";
import { ensureMedicalConfiguration } from "./configuration.js";

export class PatientQuestionAgent {
  constructor(
    private loadChatModel: (modelName: string) => Promise<any>
  ) {}

  async generateQuestions(
    state: MedicalDiagnosticStateType,
    config: RunnableConfig
  ): Promise<PatientQuestion[]> {
    const configuration = ensureMedicalConfiguration(config);
    const model = await this.loadChatModel(configuration.model);

    const questionPrompt = this.buildQuestionPrompt(state);
    const response = await model.invoke([new HumanMessage(questionPrompt)]);
    
    const questions = this.parseQuestions(response.content as string, state);
    return this.prioritizeQuestions(questions, state);
  }

  private buildQuestionPrompt(state: MedicalDiagnosticStateType): string {
    const currentDiagnoses = state.differentialDiagnoses
      .slice(0, 5)
      .map(d => `${d.condition} (${d.probability}%)`)
      .join(', ');

    const availableInfo = this.summarizeAvailableInfo(state);
    const recentAgentRecommendations = this.getRecentRecommendations(state);

    return `
You are a medical information specialist. Based on the current diagnostic state, generate 3-5 targeted follow-up questions to help clarify the diagnosis.

CURRENT SITUATION:
Chief Complaint: ${state.availableCaseInfo.chiefComplaint}
Top Diagnoses: ${currentDiagnoses}
Confidence Level: ${(state.confidenceLevel * 100).toFixed(1)}%
Current Phase: ${state.currentPhase}

AVAILABLE INFORMATION:
${availableInfo}

RECENT AGENT RECOMMENDATIONS:
${recentAgentRecommendations}

QUESTION GENERATION GUIDELINES:
1. Focus on information gaps that would help discriminate between top diagnoses
2. Ask about specific symptoms, timing, severity, and context
3. Include relevant medical history, family history, or medication questions
4. Prioritize questions that would have the highest diagnostic value
5. Avoid asking about information already provided

Generate questions in this JSON format:
{
  "questions": [
    {
      "category": "symptoms|history|examination|family|social|medications|allergies",
      "question": "Clear, specific question text",
      "priority": "high|medium|low",
      "reasoning": "Why this question helps with differential diagnosis"
    }
  ]
}

Focus on the most diagnostically valuable questions. Maximum 5 questions.
`;
  }

  private summarizeAvailableInfo(state: MedicalDiagnosticStateType): string {
    const info = state.availableCaseInfo;
    const summary = [
      `Age: ${info.demographics.age}, Gender: ${info.demographics.gender}`
    ];

    if (info.historyOfPresentIllness) {
      summary.push(`History: ${info.historyOfPresentIllness.substring(0, 200)}...`);
    }

    if (info.pastMedicalHistory?.length) {
      summary.push(`Past Medical History: ${info.pastMedicalHistory.join(', ')}`);
    }

    if (info.medications?.length) {
      summary.push(`Medications: ${info.medications.join(', ')}`);
    }

    if (info.familyHistory?.length) {
      summary.push(`Family History: ${info.familyHistory.join(', ')}`);
    }

    if (state.diagnosticTests.length > 0) {
      summary.push(`Tests: ${state.diagnosticTests.map(t => `${t.testName}: ${t.result}`).join(', ')}`);
    }

    return summary.join('\n');
  }

  private getRecentRecommendations(state: MedicalDiagnosticStateType): string {
    const recentTurns = state.agentTurns.slice(-3);
    return recentTurns
      .map(turn => `${turn.agentRole}: ${turn.recommendations.join('; ')}`)
      .join('\n');
  }

  private parseQuestions(content: string, state: MedicalDiagnosticStateType): PatientQuestion[] {
    try {
      const parsed = JSON.parse(content);
      
      return parsed.questions.map((q: any, index: number) => ({
        id: `q_${Date.now()}_${index}`,
        requestingAgent: 'hypothesis' as const,
        category: q.category,
        question: q.question,
        priority: q.priority,
        timestamp: new Date()
      }));
    } catch (error) {
      // Fallback: generate default questions based on current state
      return this.generateDefaultQuestions(state);
    }
  }

  private generateDefaultQuestions(state: MedicalDiagnosticStateType): PatientQuestion[] {
    const questions: PatientQuestion[] = [];
    
    // Always ask about symptom timeline if not clear
    if (!state.availableCaseInfo.historyOfPresentIllness?.includes('started') && 
        !state.availableCaseInfo.historyOfPresentIllness?.includes('began')) {
      questions.push({
        id: `q_${Date.now()}_timeline`,
        requestingAgent: 'hypothesis',
        category: 'symptoms',
        question: 'When did your symptoms first begin, and how have they changed over time?',
        priority: 'high',
        timestamp: new Date()
      });
    }

    // Ask about pain characteristics if pain is mentioned
    if (state.availableCaseInfo.chiefComplaint.toLowerCase().includes('pain')) {
      questions.push({
        id: `q_${Date.now()}_pain`,
        requestingAgent: 'hypothesis',
        category: 'symptoms',
        question: 'On a scale of 1-10, how would you rate your pain, and what makes it better or worse?',
        priority: 'high',
        timestamp: new Date()
      });
    }

    // Ask about medications if none listed
    if (!state.availableCaseInfo.medications?.length) {
      questions.push({
        id: `q_${Date.now()}_meds`,
        requestingAgent: 'hypothesis',
        category: 'medications',
        question: 'Are you currently taking any medications, including over-the-counter drugs or supplements?',
        priority: 'medium',
        timestamp: new Date()
      });
    }

    return questions;
  }

  private prioritizeQuestions(questions: PatientQuestion[], state: MedicalDiagnosticStateType): PatientQuestion[] {
    // Score questions based on diagnostic value
    const scoredQuestions = questions.map(q => ({
      ...q,
      score: this.calculateQuestionScore(q, state)
    }));

    // Sort by score (descending) and return top 5
    return scoredQuestions
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(({ score, ...question }) => question);
  }

  private calculateQuestionScore(question: PatientQuestion, state: MedicalDiagnosticStateType): number {
    let score = 0;

    // Priority weighting
    switch (question.priority) {
      case 'high': score += 3; break;
      case 'medium': score += 2; break;
      case 'low': score += 1; break;
    }

    // Category weighting based on current phase
    switch (state.currentPhase) {
      case 'initial_assessment':
        if (question.category === 'symptoms' || question.category === 'history') score += 2;
        break;
      case 'information_gathering':
        if (question.category === 'family' || question.category === 'social') score += 2;
        break;
      case 'patient_interaction':
        score += 1; // All questions valuable in this phase
        break;
    }

    // Boost score if question relates to high-probability diagnoses
    if (state.differentialDiagnoses.length > 0) {
      const topDiagnosis = state.differentialDiagnoses[0];
      if (topDiagnosis.probability > 60) {
        score += 2;
      }
    }

    return score;
  }

  shouldGenerateQuestions(state: MedicalDiagnosticStateType): boolean {
    // Check if we need more information
    if (state.confidenceLevel < 0.6) return true;
    
    // Check if we have too many competing diagnoses
    if (state.differentialDiagnoses.length > 4) return true;
    
    // Check if agents are requesting information
    const recentTurns = state.agentTurns.slice(-2);
    const hasInfoRequests = recentTurns.some(turn => 
      turn.recommendations.some(rec => 
        rec.toLowerCase().includes('need') || 
        rec.toLowerCase().includes('require') ||
        rec.toLowerCase().includes('ask') ||
        rec.toLowerCase().includes('clarify')
      )
    );
    
    if (hasInfoRequests) return true;
    
    // Check if we're in information gathering phase
    if (state.currentPhase === 'information_gathering') return true;
    
    // Don't generate questions if we already have pending ones
    if (state.pendingQuestions.length > 0) return false;
    
    return false;
  }
}