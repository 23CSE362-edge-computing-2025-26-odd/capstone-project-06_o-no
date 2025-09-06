package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppModule;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.fog.scheduler.TupleScheduler;
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

public class CloudMLModule extends AppModule {
    private static List<Map<String, Object>> accumulatedData = new ArrayList<>();
    private int hostDeviceId;

    public CloudMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new TupleScheduler(mips, 1), new HashMap<>());
        this.hostDeviceId = hostDeviceId;
    }

    protected void processTupleArrival(Tuple tuple) {
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            accumulatedData.add(data);
            System.out.println("Cloud received data, total: " + accumulatedData.size());

            // Predict with RF
            String tempFile = "python_ml/temp_input_" + UUID.randomUUID() + ".json";
            JSONObject json = new JSONObject(data);
            try (FileWriter fw = new FileWriter(tempFile)) {
                fw.write(json.toJSONString());
            } catch (Exception e) {
                e.printStackTrace();
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("python", "python_ml/predict_rf.py", tempFile);
                Process p = pb.start();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String output = br.readLine();
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = (double) result.get("prob");
                int fault = ((Long) result.get("fault")).intValue();

                System.out.println("Cloud RF prediction: fault=" + fault + ", prob=" + prob);

                if (fault == 1) {
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0,
                            new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP");
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Periodically train and update
            if (accumulatedData.size() % 50 == 0) {
                String accFile = "python_ml/accumulated.csv";
                try (FileWriter fw = new FileWriter(accFile)) {
                    fw.write("vibration,temp,voltage\n");
                    for (Map<String, Object> entry : accumulatedData) {
                        @SuppressWarnings("unchecked")
                        List<Double> vibration = (List<Double>) entry.get("vibration");
                        Double temp = (Double) entry.get("temp");
                        Double voltage = (Double) entry.get("voltage");
                        fw.write(String.format("%s,%f,%f\n", vibration.toString(), temp, voltage));
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                try {
                    ProcessBuilder pbRf = new ProcessBuilder("python", "python_ml/train_rf.py", accFile);
                    pbRf.start().waitFor();
                } catch (Exception e) {
                    e.printStackTrace();
                }

                try {
                    ProcessBuilder pbCnn = new ProcessBuilder("python", "python_ml/train_cnn.py", accFile);
                    pbCnn.start().waitFor();
                } catch (Exception e) {
                    e.printStackTrace();
                }

                File modelFile = new File("python_ml/model.h5");
                byte[] bytes = new byte[(int) modelFile.length()];
                try (FileInputStream fis = new FileInputStream(modelFile)) {
                    fis.read(bytes);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                String base64 = Base64.getEncoder().encodeToString(bytes);

                DataTuple updateTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.DOWN, 1000000, 1, bytes.length, 0,
                        new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                updateTuple.setTupleType("UPDATE_MODEL");
                Map<String, Object> updatePayload = new HashMap<>();
                updatePayload.put("model_base64", base64);
                updateTuple.setPayload(updatePayload);
                sendTuple(updateTuple, "EdgeML");
                System.out.println("Sent model update to edge");
            }
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}