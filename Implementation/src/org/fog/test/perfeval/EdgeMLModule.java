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
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.io.File;
import java.util.logging.Logger;

/**
 * EdgeML Module: Handles CNN predictions on edge and model updates from cloud.
 */
public class EdgeMLModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(EdgeMLModule.class.getName());
    private String modelPath = IntelliPdM.projectDirPath + "/python_ml/model.h5";
    private int hostDeviceId;

    public EdgeMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new CustomTupleScheduler(mips, 1), new HashMap<>());
        this.hostDeviceId = hostDeviceId;
    }

    protected void processTupleArrival(Tuple tuple) {
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            if ("UPDATE_MODEL".equals(tuple.getTupleType())) {
                String base64Model = (String) data.get("model_base64");
                byte[] modelBytes = java.util.Base64.getDecoder().decode(base64Model);
                try (FileOutputStream fos = new FileOutputStream(modelPath)) {
                    fos.write(modelBytes);
                    LOGGER.info("Edge model updated at time " + CloudSim.clock());
                } catch (Exception e) {
                    LOGGER.severe("Failed to update edge model: " + e.getMessage());
                }
                return;
            }

            // Preprocessed data prediction
            String tempFile = IntelliPdM.projectDirPath + "/python_ml/temp_input_" + UUID.randomUUID() + ".json";
            JSONObject json = new JSONObject();
            for (Map.Entry<String, Object> entry : data.entrySet()) {
                json.put(entry.getKey(), entry.getValue());
            }
            try (FileWriter fw = new FileWriter(tempFile)) {
                fw.write(json.toJSONString());
            } catch (Exception e) {
                LOGGER.severe("Failed to write temp input: " + e.getMessage());
            }

            try {
                ProcessBuilder pb = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/predict_cnn.py", tempFile);
                pb.directory(new File(IntelliPdM.projectDirPath));
                Process p = pb.start();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String output = br.readLine();
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = ((Number) result.get("prob")).doubleValue();
                int fault = ((Number) result.get("fault")).intValue();

                int trueFault = ((Number) data.getOrDefault("true_fault", 0)).intValue();
                LOGGER.info("Edge CNN prediction at time " + CloudSim.clock() + ": fault=" + fault + ", prob=" + prob + ", true_fault=" + trueFault);
                MetricsCollector.recordEdgePrediction(fault, trueFault);

                if (fault == 1) {
                    int machineId = ((Number) data.get("machine_id")).intValue();
                    String actuatorName = "actuator-" + machineId;
                    Actuator actuator = IntelliPdM.actuators.stream().filter(a -> a.getName().equals(actuatorName)).findFirst().get();
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP_ACTUATOR");
                    stopTuple.setActuatorId(actuator.getId());
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } catch (Exception e) {
                LOGGER.severe("CNN prediction failed: " + e.getMessage());
                // Fallback: Simple threshold fault detection
                double temp = ((Number) data.get("temp")).doubleValue();
                int fallbackFault = (temp > 60) ? 1 : 0;
                LOGGER.warning("Fallback fault detection: " + fallbackFault);
                if (fallbackFault == 1) {
                    int machineId = ((Number) data.get("machine_id")).intValue();
                    String actuatorName = "actuator-" + machineId;
                    Actuator actuator = IntelliPdM.actuators.stream().filter(a -> a.getName().equals(actuatorName)).findFirst().get();
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP_ACTUATOR");
                    stopTuple.setActuatorId(actuator.getId());
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } finally {
                new File(tempFile).delete(); // Clean up
            }
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}