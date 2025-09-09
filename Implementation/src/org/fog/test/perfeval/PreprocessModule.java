package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppModule;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.GeoLocation;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Preprocess Module: Handles missing data and forwards to EdgeML/CloudML.
 */
public class PreprocessModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(PreprocessModule.class.getName());
    private int hostDeviceId;

    public PreprocessModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new CustomTupleScheduler(mips, 1), new HashMap<>());
        this.hostDeviceId = hostDeviceId;
    }

    protected void processTupleArrival(Tuple tuple) {
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            // Fill missing values
            @SuppressWarnings("unchecked")
            List<Double> vibration = (List<Double>) data.get("vibration");
            double sum = 0, count = 0;
            for (Double v : vibration) {
                if (Double.isNaN(v)) continue;
                sum += v;
                count++;
            }
            double mean = (count > 0) ? sum / count : 0.0;
            for (int i = 0; i < vibration.size(); i++) {
                if (Double.isNaN(vibration.get(i))) vibration.set(i, mean);
            }
            if (!data.containsKey("temp") || Double.isNaN((Double) data.get("temp"))) data.put("temp", 50.0);
            if (!data.containsKey("voltage") || Double.isNaN((Double) data.get("voltage"))) data.put("voltage", 220.0);

            LOGGER.info("Preprocessed data at time " + CloudSim.clock() + ": filled missing values.");

            // Forward to EdgeML
            DataTuple edgeTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 1000, 1000,
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            edgeTuple.setPayload(new HashMap<>(data));
            edgeTuple.setTupleType("PROCESSED_TO_EDGE");
            edgeTuple.setDestModuleName("EdgeML");
            sendTuple(edgeTuple, "EdgeML");

            // Forward to CloudML
            DataTuple cloudTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 1000, 1000,
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            cloudTuple.setPayload(new HashMap<>(data));
            cloudTuple.setTupleType("PROCESSED_TO_CLOUD");
            cloudTuple.setDestModuleName("CloudML");
            sendTuple(cloudTuple, "CloudML");
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}