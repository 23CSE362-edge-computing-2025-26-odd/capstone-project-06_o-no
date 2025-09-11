package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppModule;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.GeoLocation;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.Base64;
import java.io.File;
import java.io.FileInputStream;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.util.logging.Logger;


public class CloudMLModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(CloudMLModule.class.getName());
    private String modelPath = IntelliPdM.projectDirPath + "/python_ml/rf_model.pkl";
    private int hostDeviceId;
    private double lastModelUpdateTime = 0.0;
    private static final double MODEL_UPDATE_INTERVAL = 100.0; 

    public CloudMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, CustomTupleScheduler scheduler, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", scheduler, new HashMap<>());
        this.hostDeviceId = hostDeviceId;
        ((CustomTupleScheduler) getCloudletScheduler()).setModule(this);
    }


    protected void processTupleArrival(Tuple tuple) {
        double startTime = CloudSim.clock();
        LOGGER.info("CloudMLModule received tuple at time " + String.format("%.2f", CloudSim.clock()) + ": " + tuple.getTupleType());
        
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
            
            LOGGER.info("CloudML processing data for Machine-" + machineId + " at time " + String.format("%.2f", CloudSim.clock()));
            handlePrediction(data, startTime);
            
            checkAndSendModelUpdates();
        }
    }
    
    private void handlePrediction(Map<String, Object> data, double startTime) {
        int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
        int trueFault = ((Number) data.getOrDefault("true_fault", 0)).intValue();
        double temp = ((Number) data.getOrDefault("temp", 50.0)).doubleValue();
        double voltage = ((Number) data.getOrDefault("voltage", 220.0)).doubleValue();
        
        LOGGER.info("CloudML analyzing Machine-" + machineId + 
                   " (temp=" + String.format("%.1f", temp) + 
                   "C, voltage=" + String.format("%.1f", voltage) + 
                   "V, true_fault=" + trueFault + ")");

        String tempFile = IntelliPdM.projectDirPath + "/python_ml/temp_input_cloud_" + UUID.randomUUID() + ".json";
        JSONObject json = new JSONObject();
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            json.put(entry.getKey(), entry.getValue());
        }
        
        try (FileWriter fw = new FileWriter(tempFile)) {
            fw.write(json.toJSONString());
        } catch (Exception e) {
            LOGGER.severe("Failed to write temp input: " + e.getMessage());
            return;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/predict_rf.py", tempFile);
            pb.directory(new File(IntelliPdM.projectDirPath));
            Process p = pb.start();
            BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String output = br.readLine();
            
            if (output != null) {
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = ((Number) result.get("prob")).doubleValue();
                int fault = ((Number) result.get("fault")).intValue();
                double latency = ((Number) result.getOrDefault("latency_ms", 0.0)).doubleValue();
                String method = (String) result.getOrDefault("method", "random_forest");

                double totalLatency = (CloudSim.clock() - startTime) * 1000; 
                String cloudAdvantage = "";
                if (fault == 1 && trueFault == 1) {
                    cloudAdvantage = " [CLOUD DETECTION]";
                }
                
                LOGGER.info("CloudML Prediction for Machine-" + machineId + 
                           ": FAULT=" + (fault == 1 ? "YES" : "NO") + 
                           " (probability=" + String.format("%.3f", prob) + 
                           ", method=" + method + 
                           ", latency=" + String.format("%.2f", latency) + "ms)" +
                           " | Expected: " + (trueFault == 1 ? "FAULT" : "NORMAL") + cloudAdvantage);

                MetricsCollector.recordCloudPrediction(fault, trueFault, latency, method, machineId);
                MetricsCollector.updateNetworkUsage(2000); 

                if (fault == 1) {
                    triggerActuator(machineId, prob, method);
                }
            }
        } catch (Exception e) {
            LOGGER.severe("RF prediction failed: " + e.getMessage());
            handleFallbackPrediction(data, machineId, trueFault, startTime);
        } finally {
            new File(tempFile).delete(); 
        }
    }
    
    private void handleFallbackPrediction(Map<String, Object> data, int machineId, int trueFault, double startTime) {
        double temp = ((Number) data.get("temp")).doubleValue();
        double voltage = ((Number) data.get("voltage")).doubleValue();
        
        int fallbackFault = (temp > 70 || voltage > 260 || voltage < 180) ? 1 : 0;
        double prob = fallbackFault == 1 ? 0.9 : 0.1;
        double latency = (CloudSim.clock() - startTime) * 1000;
        
        LOGGER.warning("CloudML Advanced Fallback Prediction for Machine-" + machineId + 
                      ": FAULT=" + (fallbackFault == 1 ? "YES" : "NO") + 
                      " (advanced-threshold, latency=" + String.format("%.2f", latency) + "ms)");
        
        MetricsCollector.recordCloudPrediction(fallbackFault, trueFault, latency, "fallback_advanced", machineId);
        
        if (fallbackFault == 1) {
            triggerActuator(machineId, prob, "fallback_advanced");
        }
    }
    
    private void triggerActuator(int machineId, double probability, String method) {
        String actuatorName = "actuator-" + machineId;
        try {
            Actuator actuator = IntelliPdM.actuators.stream()
                .filter(a -> a.getName().equals(actuatorName))
                .findFirst().get();
            
            DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0, 
                new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            stopTuple.setUserId(getUserId());
            stopTuple.setTupleType("STOP_ACTUATOR");
            stopTuple.setActuatorId(actuator.getId());
            sendTuple(stopTuple, "STOP_ACTUATOR");
            
            LOGGER.info("CloudML triggered STOP for Machine-" + machineId + 
                       " (probability=" + String.format("%.3f", probability) + 
                       ", method=" + method + ") at time " + String.format("%.2f", CloudSim.clock()));
                       
        } catch (Exception e) {
            LOGGER.severe("Failed to trigger actuator for Machine-" + machineId + ": " + e.getMessage());
        }
    }
    
    private void checkAndSendModelUpdates() {
        double currentTime = CloudSim.clock();
        if (currentTime - lastModelUpdateTime >= MODEL_UPDATE_INTERVAL) {
            sendModelUpdateToEdges();
            lastModelUpdateTime = currentTime;
        }
    }
    
    private void sendModelUpdateToEdges() {
                    LOGGER.info("CloudML initiating model update to edge devices at time " + String.format("%.2f", CloudSim.clock()));
        
        try {
            File modelFile = new File(IntelliPdM.projectDirPath + "/python_ml/ann_model.keras");
            if (!modelFile.exists()) {
                LOGGER.warning(" ANN model not found, skipping model update");
                return;
            }
            
            byte[] modelBytes = new byte[(int) modelFile.length()];
            try (FileInputStream fis = new FileInputStream(modelFile)) {
                fis.read(modelBytes);
            }
            String base64Model = Base64.getEncoder().encodeToString(modelBytes);
            
            Map<String, Object> updateData = new HashMap<>();
            updateData.put("model_base64", base64Model);
            updateData.put("update_time", CloudSim.clock());
            updateData.put("model_version", System.currentTimeMillis());
            
            DataTuple updateTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.DOWN, 1000000, 1, 1000000, 1000,
                new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            updateTuple.setUserId(getUserId());
            updateTuple.setPayload(updateData);
            updateTuple.setTupleType("UPDATE_MODEL");
            updateTuple.setDestModuleName("EdgeML");
            
            for (FogDevice device : IntelliPdM.fogDevices) {
                if (device.getName().startsWith("edge")) {
                    sendTuple(updateTuple, "EdgeML");
                    LOGGER.info("Model update sent to " + device.getName() + " at time " + String.format("%.2f", CloudSim.clock()));
                }
            }
            
            MetricsCollector.updateNetworkUsage(1000000); 
        } catch (Exception e) {
            LOGGER.severe("Failed to send model update: " + e.getMessage());
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        
        tuple.setDestModuleName(destModule);
        
        CloudSim.send(hostDeviceId, hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
        LOGGER.info("CloudMLModule sent tuple to " + destModule + " at time " + String.format("%.2f", CloudSim.clock()));
    }
}