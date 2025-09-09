package org.fog.test.perfeval;

import org.fog.entities.FogDevice;
import java.util.List;

/**
 * Utility class to collect and store simulation metrics.
 */
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

    public static void incrementStoppedMachines() {
        stoppedMachines++;
    }

    public static void updateNetworkUsage() {
        // Placeholder: Implement actual network usage calculation if needed
        totalNetworkUsage += 100.0; // Example increment
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

    public static void recordEdgePrediction(int predictedFault, int trueFault) {
        edgeTotalPredictions++;
        if (predictedFault == trueFault) {
            edgeCorrectPredictions++;
        }
        if (predictedFault == 1) {
            incrementEdgeFaults();
        }
        if (edgeTotalPredictions > 0) {
            setEdgeAccuracy((double) edgeCorrectPredictions / (double) edgeTotalPredictions);
        }
    }

    public static void recordCloudPrediction(int predictedFault, int trueFault) {
        cloudTotalPredictions++;
        if (predictedFault == trueFault) {
            cloudCorrectPredictions++;
        }
        if (predictedFault == 1) {
            incrementCloudFaults();
        }
        if (cloudTotalPredictions > 0) {
            setCloudAccuracy((double) cloudCorrectPredictions / (double) cloudTotalPredictions);
        }
    }
}