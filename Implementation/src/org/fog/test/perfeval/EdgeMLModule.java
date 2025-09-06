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
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class EdgeMLModule extends AppModule {
    private String modelPath = "python_ml/model.h5";
    private int hostDeviceId;

    public EdgeMLModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new TupleScheduler(mips, 1), new HashMap<>());
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
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println("Edge model updated");
                return;
            }

            String tempFile = "python_ml/temp_input_" + UUID.randomUUID() + ".json";
            JSONObject json = new JSONObject(data);
            try (FileWriter fw = new FileWriter(tempFile)) {
                fw.write(json.toJSONString());
            } catch (Exception e) {
                e.printStackTrace();
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("python", "python_ml/predict_cnn.py", tempFile);
                Process p = pb.start();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String output = br.readLine();
                JSONParser parser = new JSONParser();
                JSONObject result = (JSONObject) parser.parse(output);
                double prob = (double) result.get("prob");
                int fault = ((Long) result.get("fault")).intValue();

                System.out.println("Edge CNN prediction: fault=" + fault + ", prob=" + prob);

                if (fault == 1) {
                    DataTuple stopTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.ACTUATOR, 100, 1, 100, 0, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
                    stopTuple.setTupleType("STOP");
                    sendTuple(stopTuple, "STOP_ACTUATOR");
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}