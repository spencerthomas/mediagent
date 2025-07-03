/**
 * Medical Diagnostic Tools
 * Tools for medical information retrieval, test selection, and cost analysis
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { CaseInformationGatekeeper, InformationRequest } from "./gatekeeper.js";

// Medical knowledge base for diagnostic information
const MEDICAL_KNOWLEDGE_BASE: {
  conditions: Record<string, {
    symptoms: string[];
    tests: string[];
    prevalence: number;
    icd10: string;
  }>;
  diagnosticTests: Record<string, {
    cost: number;
    sensitivity: number;
    specificity: number;
    turnaround: string;
  }>;
} = {
  conditions: {
    'myocardial_infarction': {
      symptoms: ['chest pain', 'shortness of breath', 'nausea', 'diaphoresis'],
      tests: ['ECG', 'troponin', 'chest x-ray', 'echocardiogram'],
      prevalence: 0.02,
      icd10: 'I21.9'
    },
    'pneumonia': {
      symptoms: ['fever', 'cough', 'shortness of breath', 'chest pain'],
      tests: ['chest x-ray', 'CBC', 'blood culture', 'sputum culture'],
      prevalence: 0.05,
      icd10: 'J18.9'
    },
    'gastroenteritis': {
      symptoms: ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],
      tests: ['stool culture', 'CBC', 'basic metabolic panel'],
      prevalence: 0.15,
      icd10: 'K59.1'
    }
  },
  
  diagnosticTests: {
    'ECG': { cost: 50, sensitivity: 0.85, specificity: 0.90, turnaround: '5 minutes' },
    'troponin': { cost: 75, sensitivity: 0.95, specificity: 0.85, turnaround: '2 hours' },
    'chest_xray': { cost: 150, sensitivity: 0.70, specificity: 0.80, turnaround: '1 hour' },
    'cbc': { cost: 25, sensitivity: 0.60, specificity: 0.70, turnaround: '2 hours' },
    'bmp': { cost: 30, sensitivity: 0.65, specificity: 0.75, turnaround: '2 hours' }
  }
};

/**
 * Tool for requesting medical information through the gatekeeper
 */
export const createInformationRequestTool = (_gatekeeper: CaseInformationGatekeeper) => 
  new DynamicStructuredTool({
    name: "request_medical_information",
    description: "Request specific medical information about the patient through the gatekeeper system",
    schema: z.object({
      requestType: z.enum(['history', 'physical_exam', 'lab_results', 'imaging', 'specialist_consultation']),
      specificRequest: z.string().describe("Specific information or test being requested"),
      justification: z.string().describe("Clinical justification for the request"),
      requestedBy: z.string().describe("Agent role making the request")
    }),
    func: async ({ requestType, specificRequest, justification, requestedBy }) => {
      const request: InformationRequest = {
        requestedBy,
        requestType,
        specificRequest,
        justification,
        estimatedCost: estimateRequestCost(requestType)
      };

      // This would be called with the current state in a real implementation
      // For now, return a mock response
      return `Information request submitted: ${specificRequest}. Estimated cost: $${request.estimatedCost}`;
    }
  });

/**
 * Tool for diagnostic test selection and recommendation
 */
export const diagnosticTestSelectorTool = new DynamicStructuredTool({
  name: "diagnostic_test_selector",
  description: "Get recommendations for diagnostic tests based on differential diagnoses",
  schema: z.object({
    differentialDiagnoses: z.array(z.string()).describe("List of diagnostic possibilities to evaluate"),
    patientAge: z.number().describe("Patient age"),
    patientGender: z.enum(['male', 'female']).describe("Patient gender"),
    chiefComplaint: z.string().describe("Patient's chief complaint"),
    maxCost: z.number().optional().describe("Maximum cost constraint for testing")
  }),
  func: async ({ differentialDiagnoses, maxCost }) => {
    const recommendations = [];
    
    for (const diagnosis of differentialDiagnoses) {
      const normalizedDx = diagnosis.toLowerCase().replace(/\s+/g, '_');
      const condition = MEDICAL_KNOWLEDGE_BASE.conditions[normalizedDx];
      
      if (condition) {
        const recommendedTests = condition.tests.map((test: string) => {
          const testInfo = MEDICAL_KNOWLEDGE_BASE.diagnosticTests[test.toLowerCase().replace(/\s+/g, '_')];
          return {
            test,
            cost: testInfo?.cost || 100,
            sensitivity: testInfo?.sensitivity || 0.8,
            specificity: testInfo?.specificity || 0.8,
            turnaround: testInfo?.turnaround || '4 hours'
          };
        });
        
        recommendations.push({
          diagnosis,
          recommendedTests: recommendedTests.filter(t => !maxCost || t.cost <= maxCost)
        });
      }
    }
    
    return JSON.stringify(recommendations, null, 2);
  }
});

/**
 * Tool for cost estimation and budget tracking
 */
export const costEstimatorTool = new DynamicStructuredTool({
  name: "cost_estimator",
  description: "Calculate costs for diagnostic tests and track budget utilization",
  schema: z.object({
    tests: z.array(z.string()).describe("List of diagnostic tests to cost"),
    currentBudget: z.number().describe("Current available budget"),
    cumulativeCost: z.number().describe("Already spent amount")
  }),
  func: async ({ tests, currentBudget, cumulativeCost }) => {
    const costAnalysis: {
      testCosts: any[];
      totalCost: number;
      remainingBudget: number;
      budgetUtilization: number;
      recommendations: string[];
    } = {
      testCosts: [],
      totalCost: 0,
      remainingBudget: currentBudget - cumulativeCost,
      budgetUtilization: 0,
      recommendations: []
    };

    for (const test of tests) {
      const normalizedTest = test.toLowerCase().replace(/\s+/g, '_');
      const testInfo = MEDICAL_KNOWLEDGE_BASE.diagnosticTests[normalizedTest];
      const cost = testInfo?.cost || estimateTestCost(test);
      
      costAnalysis.testCosts.push({
        test,
        cost,
        diagnosticValue: testInfo?.sensitivity || 0.8
      });
      
      costAnalysis.totalCost += cost;
    }

    costAnalysis.budgetUtilization = (cumulativeCost + costAnalysis.totalCost) / currentBudget;
    
    // Generate recommendations
    if (costAnalysis.budgetUtilization > 0.8) {
      costAnalysis.recommendations.push("Consider prioritizing highest-yield tests to stay within budget");
    }
    
    if (costAnalysis.totalCost > costAnalysis.remainingBudget) {
      costAnalysis.recommendations.push("Proposed tests exceed remaining budget - consider alternatives");
    }

    return JSON.stringify(costAnalysis, null, 2);
  }
});

/**
 * Tool for differential diagnosis ranking
 */
export const differentialDiagnosisRankerTool = new DynamicStructuredTool({
  name: "differential_diagnosis_ranker",
  description: "Rank differential diagnoses based on clinical presentation and test results",
  schema: z.object({
    symptoms: z.array(z.string()).describe("Patient symptoms"),
    testResults: z.array(z.object({
      test: z.string(),
      result: z.string(),
      abnormal: z.boolean()
    })).describe("Available test results"),
    patientAge: z.number().describe("Patient age"),
    patientGender: z.enum(['male', 'female']).describe("Patient gender")
  }),
  func: async ({ symptoms, testResults }) => {
    const rankings = [];
    
    for (const [condition, info] of Object.entries(MEDICAL_KNOWLEDGE_BASE.conditions)) {
      let score = 0;
      let maxScore = 0;
      
      // Score based on symptom match
      for (const symptom of symptoms) {
        maxScore += 10;
        if (info.symptoms.some(s => s.toLowerCase().includes(symptom.toLowerCase()))) {
          score += 10;
        }
      }
      
      // Score based on test results
      for (const testResult of testResults) {
        if (info.tests.includes(testResult.test) && testResult.abnormal) {
          score += 15;
        }
        maxScore += 15;
      }
      
      // Adjust for prevalence
      score *= info.prevalence * 100;
      
      const probability = maxScore > 0 ? Math.min((score / maxScore) * 100, 100) : 0;
      
      rankings.push({
        condition: condition.replace(/_/g, ' '),
        probability: Math.round(probability),
        icd10: info.icd10,
        supportingFactors: info.symptoms.filter(s => 
          symptoms.some(symptom => s.toLowerCase().includes(symptom.toLowerCase()))
        )
      });
    }
    
    // Sort by probability
    rankings.sort((a, b) => b.probability - a.probability);
    
    return JSON.stringify(rankings, null, 2);
  }
});

/**
 * Tool for medical knowledge base queries
 */
export const medicalKnowledgeBaseTool = new DynamicStructuredTool({
  name: "medical_knowledge_base",
  description: "Query medical knowledge base for condition information, test characteristics, and clinical guidelines",
  schema: z.object({
    query: z.string().describe("Medical query or condition name"),
    queryType: z.enum(['condition', 'test', 'guideline']).describe("Type of information requested")
  }),
  func: async ({ query, queryType }) => {
    const normalizedQuery = query.toLowerCase().replace(/\s+/g, '_');
    
    switch (queryType) {
      case 'condition':
        const condition = MEDICAL_KNOWLEDGE_BASE.conditions[normalizedQuery];
        if (condition) {
          return JSON.stringify({
            condition: query,
            symptoms: condition.symptoms,
            recommendedTests: condition.tests,
            prevalence: condition.prevalence,
            icd10: condition.icd10
          }, null, 2);
        }
        return `No information found for condition: ${query}`;
      
      case 'test':
        const test = MEDICAL_KNOWLEDGE_BASE.diagnosticTests[normalizedQuery];
        if (test) {
          return JSON.stringify({
            test: query,
            cost: test.cost,
            sensitivity: test.sensitivity,
            specificity: test.specificity,
            turnaround: test.turnaround
          }, null, 2);
        }
        return `No information found for test: ${query}`;
      
      default:
        return `Query type ${queryType} not implemented`;
    }
  }
});

/**
 * Tool for bias detection in diagnostic reasoning
 */
export const biasDetectorTool = new DynamicStructuredTool({
  name: "bias_detector",
  description: "Identify potential cognitive biases in diagnostic reasoning",
  schema: z.object({
    diagnosticReasoning: z.string().describe("The diagnostic reasoning to analyze"),
    differentialDiagnoses: z.array(z.string()).describe("Current differential diagnoses"),
    initialImpression: z.string().describe("Initial diagnostic impression")
  }),
  func: async ({ diagnosticReasoning, differentialDiagnoses }) => {
    const biases = [];
    
    // Check for anchoring bias
    if (differentialDiagnoses.length <= 2) {
      biases.push({
        type: 'anchoring_bias',
        description: 'Limited differential diagnosis suggests potential anchoring on initial impression',
        severity: 'moderate'
      });
    }
    
    // Check for confirmation bias
    if (diagnosticReasoning.toLowerCase().includes('confirm') || 
        diagnosticReasoning.toLowerCase().includes('rule in')) {
      biases.push({
        type: 'confirmation_bias',
        description: 'Language suggests seeking confirmation rather than ruling out alternatives',
        severity: 'mild'
      });
    }
    
    // Check for availability bias
    if (diagnosticReasoning.toLowerCase().includes('recent') || 
        diagnosticReasoning.toLowerCase().includes('just saw')) {
      biases.push({
        type: 'availability_bias',
        description: 'Recent case influence detected in reasoning',
        severity: 'mild'
      });
    }
    
    // Check for premature closure
    if (diagnosticReasoning.toLowerCase().includes('obvious') || 
        diagnosticReasoning.toLowerCase().includes('clearly')) {
      biases.push({
        type: 'premature_closure',
        description: 'Overconfident language suggests potential premature closure',
        severity: 'moderate'
      });
    }
    
    return JSON.stringify({
      biasesDetected: biases,
      recommendations: biases.length > 0 ? [
        'Consider expanding differential diagnosis',
        'Seek disconfirming evidence',
        'Review alternative explanations'
      ] : ['No significant biases detected']
    }, null, 2);
  }
});

// Helper functions
function estimateRequestCost(requestType: string): number {
  const baseCosts: Record<string, number> = {
    'history': 0,
    'physical_exam': 0,
    'lab_results': 50,
    'imaging': 400,
    'specialist_consultation': 300
  };
  
  return baseCosts[requestType] || 100;
}

function estimateTestCost(testName: string): number {
  const testLower = testName.toLowerCase();
  
  if (testLower.includes('blood') || testLower.includes('lab')) {
    return 50;
  } else if (testLower.includes('x-ray')) {
    return 150;
  } else if (testLower.includes('ct')) {
    return 800;
  } else if (testLower.includes('mri')) {
    return 1500;
  } else if (testLower.includes('ultrasound')) {
    return 300;
  }
  
  return 100;
}

// Export all tools
export const MEDICAL_TOOLS = [
  diagnosticTestSelectorTool,
  costEstimatorTool,
  differentialDiagnosisRankerTool,
  medicalKnowledgeBaseTool,
  biasDetectorTool
];