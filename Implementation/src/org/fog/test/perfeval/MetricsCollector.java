package org.fog.test.perfeval;

import org.fog.entities.FogDevice;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class MetricsCollector {
    private static int stoppedMachines = 0;
    private static double totalNetworkUsage = 0.0;
    private static int edgeFaults = 0;
    private static int cloudFaults = 0;
    private static double edgeAccuracy = 0.0;
    private static double cloudAccuracy = 0.0;
    private static int edgeTotalPredictions = 0;
    private static int edgeCorrectPredictions = 0;
    private static int cloudTotalPredictions = 0;
    private static int cloudCorrectPredictions = 0;
    
    private static List<Double> edgeLatencies = new ArrayList<>();
    private static List<Double> cloudLatencies = new ArrayList<>();
    private static List<Double> preprocessingLatencies = new ArrayList<>();
    private static List<Double> endToEndLatencies = new ArrayList<>();
    private static Map<String, Integer> faultsByMachine = new HashMap<>();
    private static Map<String, Integer> predictionMethods = new HashMap<>();
    private static double totalDataProcessed = 0.0;
    private static int totalSensorReadings = 0;
    private static int modelUpdatesReceived = 0;
    private static double lastModelUpdateTime = 0.0;

    public static void incrementStoppedMachines() {
        stoppedMachines++;
    }

    public static void updateNetworkUsage(double bytes) {
        totalNetworkUsage += bytes;
    }

    public static double getTotalNetworkUsage() {
        return totalNetworkUsage;
    }

    public static int getEdgeFaults() {
        return edgeFaults;
    }

    public static void incrementEdgeFaults() {
        edgeFaults++;
    }

    public static int getCloudFaults() {
        return cloudFaults;
    }

    public static void incrementCloudFaults() {
        cloudFaults++;
    }

    public static double getEdgeAccuracy() {
        return edgeAccuracy;
    }

    public static void setEdgeAccuracy(double accuracy) {
        edgeAccuracy = accuracy;
    }

    public static double getCloudAccuracy() {
        return cloudAccuracy;
    }

    public static void setCloudAccuracy(double accuracy) {
        cloudAccuracy = accuracy;
    }

    public static int getStoppedMachines() {
        return stoppedMachines;
    }

    public static double getTotalEnergy(List<FogDevice> fogDevices) {
        double totalEnergy = 0.0;
        for (FogDevice fd : fogDevices) {
            totalEnergy += fd.getEnergyConsumption();
        }
        return totalEnergy;
    }

    public static void recordEdgePrediction(int predictedFault, int trueFault, double latency, String method, int machineId) {
        edgeTotalPredictions++;
        if (predictedFault == trueFault) {
            edgeCorrectPredictions++;
        }
        if (predictedFault == 1) {
            incrementEdgeFaults();
            faultsByMachine.put("machine-" + machineId, faultsByMachine.getOrDefault("machine-" + machineId, 0) + 1);
        }
        if (edgeTotalPredictions > 0) {
            setEdgeAccuracy((double) edgeCorrectPredictions / (double) edgeTotalPredictions);
        }
        
        edgeLatencies.add(latency);
        predictionMethods.put("edge_" + method, predictionMethods.getOrDefault("edge_" + method, 0) + 1);
    }

    public static void recordCloudPrediction(int predictedFault, int trueFault, double latency, String method, int machineId) {
        cloudTotalPredictions++;
        if (predictedFault == trueFault) {
            cloudCorrectPredictions++;
        }
        if (predictedFault == 1) {
            incrementCloudFaults();
            faultsByMachine.put("machine-" + machineId, faultsByMachine.getOrDefault("machine-" + machineId, 0) + 1);
        }
        if (cloudTotalPredictions > 0) {
            setCloudAccuracy((double) cloudCorrectPredictions / (double) cloudTotalPredictions);
        }
        
        cloudLatencies.add(latency);
        predictionMethods.put("cloud_" + method, predictionMethods.getOrDefault("cloud_" + method, 0) + 1);
    }

    public static void recordEdgePrediction(int predictedFault, int trueFault) {
        recordEdgePrediction(predictedFault, trueFault, 0.0, "unknown", 0);
    }

    public static void recordCloudPrediction(int predictedFault, int trueFault) {
        recordCloudPrediction(predictedFault, trueFault, 0.0, "unknown", 0);
    }

    public static void recordPreprocessingLatency(double latency) {
        preprocessingLatencies.add(latency);
    }

    public static void recordEndToEndLatency(double latency) {
        endToEndLatencies.add(latency);
    }

    public static void recordSensorReading(double dataSize) {
        totalSensorReadings++;
        totalDataProcessed += dataSize;
    }

    public static void recordModelUpdate() {
        modelUpdatesReceived++;
        lastModelUpdateTime = org.cloudbus.cloudsim.core.CloudSim.clock();
    }

    public static double getAverageLatency(List<Double> latencies) {
        return latencies.isEmpty() ? 0.0 : latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public static double getMaxLatency(List<Double> latencies) {
        return latencies.isEmpty() ? 0.0 : latencies.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
    }

    public static double getMinLatency(List<Double> latencies) {
        return latencies.isEmpty() ? 0.0 : latencies.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    }

    public static void printDetailedMetrics() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("                    INTELLIPDM SIMULATION RESULTS");
        System.out.println("=".repeat(80));
        
        System.out.println("BASIC METRICS:");
        System.out.println("  Total Sensor Readings: " + totalSensorReadings);
        System.out.println("  Total Data Processed: " + String.format("%.2f KB", totalDataProcessed / 1024));
        System.out.println("  Machines Stopped: " + stoppedMachines);
        System.out.println("  Model Updates Received: " + modelUpdatesReceived);
        System.out.println("  Last Model Update: " + String.format("%.2f", lastModelUpdateTime));
        
        System.out.println("\nPREDICTION ACCURACY:");
        System.out.println("  Edge Predictions: " + edgeTotalPredictions + " (Accuracy: " + String.format("%.2f%%", edgeAccuracy * 100) + ")");
        System.out.println("  Cloud Predictions: " + cloudTotalPredictions + " (Accuracy: " + String.format("%.2f%%", cloudAccuracy * 100) + ")");
        System.out.println("  Edge Faults Detected: " + edgeFaults);
        System.out.println("  Cloud Faults Detected: " + cloudFaults);
        
        System.out.println("\nLATENCY METRICS (milliseconds):");
        System.out.println("  Edge Prediction Latency:");
        System.out.println("    Average: " + String.format("%.2f ms", getAverageLatency(edgeLatencies)));
        System.out.println("    Min: " + String.format("%.2f ms", getMinLatency(edgeLatencies)));
        System.out.println("    Max: " + String.format("%.2f ms", getMaxLatency(edgeLatencies)));
        
        System.out.println("  Cloud Prediction Latency:");
        System.out.println("    Average: " + String.format("%.2f ms", getAverageLatency(cloudLatencies)));
        System.out.println("    Min: " + String.format("%.2f ms", getMinLatency(cloudLatencies)));
        System.out.println("    Max: " + String.format("%.2f ms", getMaxLatency(cloudLatencies)));
        
        System.out.println("  Preprocessing Latency:");
        System.out.println("    Average: " + String.format("%.2f ms", getAverageLatency(preprocessingLatencies)));
        
        System.out.println("  End-to-End Latency:");
        System.out.println("    Average: " + String.format("%.2f ms", getAverageLatency(endToEndLatencies)));
        
        System.out.println("\nPREDICTION METHODS USED:");
        for (Map.Entry<String, Integer> entry : predictionMethods.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " predictions");
        }
        
        System.out.println("\nFAULTS BY MACHINE:");
        if (faultsByMachine.isEmpty()) {
            System.out.println("  No faults detected");
        } else {
            for (Map.Entry<String, Integer> entry : faultsByMachine.entrySet()) {
                System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " faults");
            }
        }
        
        System.out.println("\nNETWORK USAGE:");
        System.out.println("  Total Network Usage: " + String.format("%.2f KB", totalNetworkUsage / 1024));
        System.out.println("  Average per Reading: " + String.format("%.2f bytes", totalSensorReadings > 0 ? totalNetworkUsage / totalSensorReadings : 0));
        
        System.out.println("=".repeat(80));
    }
}