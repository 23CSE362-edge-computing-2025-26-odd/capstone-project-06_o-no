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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PreprocessModule extends AppModule {
    private int hostDeviceId;

    public PreprocessModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, GeoLocation geoLocation, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", new TupleScheduler(mips, 1), new HashMap<>());
        this.hostDeviceId = hostDeviceId;
    }

    protected void processTupleArrival(Tuple tuple) {
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            // Fill missing values
            List<Double> vibration = (List<Double>) data.get("vibration");
            if (vibration.contains(Double.NaN)) {
                double sum = 0, count = 0;
                for (Double v : vibration) {
                    if (!v.isNaN()) { sum += v; count++; }
                }
                double mean = sum / count;
                for (int i = 0; i < vibration.size(); i++) {
                    if (vibration.get(i).isNaN()) vibration.set(i, mean);
                }
            }
            if (!data.containsKey("temp")) data.put("temp", 50.0); 
            if (!data.containsKey("voltage")) data.put("voltage", 220.0);

            DataTuple edgeTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.DOWN, 1000, 1, 1000, 1000,
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            edgeTuple.setPayload(new HashMap<>(data));
            edgeTuple.setDestModuleName("EdgeML");
            sendTuple(edgeTuple, "EdgeML");

            DataTuple cloudTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.DOWN, 1000, 1, 1000, 1000,
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
            cloudTuple.setPayload(new HashMap<>(data));
            cloudTuple.setDestModuleName("CloudML");
            sendTuple(cloudTuple, "CloudML");
        }
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        ((FogDevice) CloudSim.getEntity(hostDeviceId)).send(hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
    }
}