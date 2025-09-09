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
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.util.Base64;
import java.util.logging.Logger;

/**
 * CloudML Module: Handles RF predictions, data accumulation, and model retraining/updates.
 */
public class CloudMLModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(CloudMLModule.class.getName());
    private static List<Map<String, Object>> accumulatedData = new ArrayList<>();
    private int hostDeviceId;

    public CloudMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new CustomTupleScheduler(mips, 1), new HashMap<>());
        this.hostDeviceId = hostDeviceId;
    }

    protected void processTupleArrival(Tuple tuple) {
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            accumulatedData.add(data);
            LOGGER.info("Cloud received data at time " + CloudSim.clock() + ", total accumulated: " + accumulatedData.size());

            // Predict with RF
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
                ProcessBuilder pb = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/predict_rf.py", tempFile);
                pb.directory(new File(IntelliPdM.projectDirPath));
                Process p = pb.start();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String output = br.readLine();
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = ((Number) result.get("prob")).doubleValue();
                int fault = ((Number) result.get("fault")).intValue();

                int trueFault = ((Number) data.getOrDefault("true_fault", 0)).intValue();
                LOGGER.info("Cloud RF prediction at time " + CloudSim.clock() + ": fault=" + fault + ", prob=" + prob + ", true_fault=" + trueFault);
                MetricsCollector.recordCloudPrediction(fault, trueFault);

                if (fault == 1) {
                    int machineId = ((Number) data.get("machine_id")).intValue();
                    String actuatorName = "actuator-" + machineId;
                    Actuator actuator = IntelliPdM.actuators.stream().filter(a -> a.getName().equals(actuatorName)).findFirst().get();
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0,
                            new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP_ACTUATOR");
                    stopTuple.setActuatorId(actuator.getId());
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } catch (Exception e) {
                LOGGER.severe("RF prediction failed: " + e.getMessage());
                // Fallback
                double voltage = ((Number) data.get("voltage")).doubleValue();
                int fallbackFault = (voltage > 230) ? 1 : 0;
                LOGGER.warning("Fallback fault detection: " + fallbackFault);
                if (fallbackFault == 1) {
                    int machineId = ((Number) data.get("machine_id")).intValue();
                    String actuatorName = "actuator-" + machineId;
                    Actuator actuator = IntelliPdM.actuators.stream().filter(a -> a.getName().equals(actuatorName)).findFirst().get();
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0,
                            new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP_ACTUATOR");
                    stopTuple.setActuatorId(actuator.getId());
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } finally {
                new File(tempFile).delete();
            }

            // Periodically train and update
            if (accumulatedData.size() % 50 == 0) {
                String accFile = IntelliPdM.projectDirPath + "/python_ml/accumulated.csv";
                try (FileWriter fw = new FileWriter(accFile)) {
                    fw.write("vibration,temp,voltage,label\n"); // Added label for training
                    for (Map<String, Object> entry : accumulatedData) {
                        @SuppressWarnings("unchecked")
                        List<Double> vibration = (List<Double>) entry.get("vibration");
                        Double temp = (Double) entry.get("temp");
                        Double voltage = (Double) entry.get("voltage");
                        int label = (int) entry.get("true_fault");
                        fw.write(String.format("%s,%f,%f,%d\n", vibration.toString(), temp, voltage, label));
                    }
                    LOGGER.info("Accumulated data written to CSV for retraining.");
                } catch (Exception e) {
                    LOGGER.severe("Failed to write accumulated CSV: " + e.getMessage());
                }

                try {
                    ProcessBuilder pbRf = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/train_rf.py", accFile);
                    pbRf.directory(new File(IntelliPdM.projectDirPath));
                    Process pRf = pbRf.start();
                    pRf.waitFor();
                    LOGGER.info("RF model retrained.");
                } catch (Exception e) {
                    LOGGER.severe("RF retraining failed: " + e.getMessage());
                }

                try {
                    ProcessBuilder pbCnn = new ProcessBuilder(IntelliPdM.pythonExec, "python_ml/train_cnn.py", accFile);
                    pbCnn.directory(new File(IntelliPdM.projectDirPath));
                    Process pCnn = pbCnn.start();
                    pCnn.waitFor();
                    LOGGER.info("CNN model retrained.");
                } catch (Exception e) {
                    LOGGER.severe("CNN retraining failed: " + e.getMessage());
                }

                File modelFile = new File(IntelliPdM.projectDirPath + "/python_ml/model.h5");
                byte[] bytes = new byte[(int) modelFile.length()];
                try (FileInputStream fis = new FileInputStream(modelFile)) {
                    fis.read(bytes);
                    String base64 = Base64.getEncoder().encodeToString(bytes);

                    DataTuple updateTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.DOWN, 1000000, 1, bytes.length, 0,
                            new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    updateTuple.setTupleType("UPDATE_MODEL");
                    Map<String, Object> updatePayload = new HashMap<>();
                    updatePayload.put("model_base64", base64);
                    updateTuple.setPayload(updatePayload);
                    updateTuple.setDestModuleName("EdgeML");
                    sendTuple(updateTuple, "EdgeML");
                    LOGGER.info("Sent model update to edge at time " + CloudSim.clock());
                } catch (Exception e) {
                    LOGGER.severe("Failed to send model update: " + e.getMessage());
                }
            }
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}