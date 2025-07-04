/**
 * Example usage of the Medical Diagnostic Agent
 * Demonstrates how to use the medical superintelligence system
 */

import { medicalGraph } from "./medical_graph.js";
import { HumanMessage } from "@langchain/core/messages";

async function runMedicalDiagnosticExample() {
  console.log("🏥 Medical Superintelligence Agent Example");
  console.log("==========================================\n");

  // Example medical case
  const medicalCase = `
Patient: 58-year-old female
Chief Complaint: Severe chest pain for 3 hours
History: Pain started suddenly while gardening, described as "crushing" and radiates to left jaw and arm. 
Associated with nausea, diaphoresis, and shortness of breath.
Past Medical History: Hypertension, diabetes mellitus type 2, hyperlipidemia
Medications: Metformin, Lisinopril, Atorvastatin
Social History: Former smoker (quit 5 years ago), occasional alcohol use
`;

  console.log("📋 Medical Case:");
  console.log(medicalCase);
  console.log("\n🤖 Starting Medical AI Diagnostic Process...\n");

  try {
    const result = await medicalGraph.invoke({
      messages: [new HumanMessage(medicalCase)]
    }, {
      configurable: {
        model: "o3-mini",
        systemPromptTemplate: "You are a medical diagnostic AI assistant. Analyze the provided medical case and provide diagnostic insights."
      }
    });

    console.log("✅ Diagnostic Process Completed!\n");
    
    console.log("📊 Results Summary:");
    console.log("===================");
    console.log(`Patient ID: ${result.availableCaseInfo.patientId}`);
    console.log(`Chief Complaint: ${result.availableCaseInfo.chiefComplaint}`);
    console.log(`Current Phase: ${result.currentPhase}`);
    console.log(`Debate Rounds: ${result.debateRound}`);
    console.log(`Total Cost: $${result.cumulativeCost}`);
    console.log(`Confidence Level: ${(result.confidenceLevel * 100).toFixed(1)}%`);
    console.log(`Ready for Diagnosis: ${result.readyForDiagnosis ? 'Yes' : 'No'}`);
    
    if (result.differentialDiagnoses.length > 0) {
      console.log("\n🔍 Differential Diagnoses:");
      result.differentialDiagnoses.slice(0, 5).forEach((dx, index) => {
        console.log(`${index + 1}. ${dx.condition} (${dx.probability}%)`);
        console.log(`   Reasoning: ${dx.reasoning.substring(0, 100)}...`);
      });
    }

    if (result.finalDiagnosis) {
      console.log("\n🎯 Final Diagnosis:");
      console.log(`${result.finalDiagnosis.condition} (${result.finalDiagnosis.probability}%)`);
      console.log(`Reasoning: ${result.finalDiagnosis.reasoning}`);
    }

    if (result.biasesDetected.length > 0) {
      console.log("\n⚠️  Biases Detected:");
      result.biasesDetected.forEach(bias => console.log(`- ${bias}`));
    }

    console.log(`\n💬 Agent Turns: ${result.agentTurns.length}`);
    console.log(`📝 Total Messages: ${result.messages.length}`);
    console.log(`🧠 Reasoning Quality: ${(result.reasoningQuality * 100).toFixed(1)}%`);

  } catch (error) {
    console.error("❌ Error running medical diagnostic agent:", error);
  }
}

// Run the example if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runMedicalDiagnosticExample();
}

export { runMedicalDiagnosticExample };