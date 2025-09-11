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


public class PreprocessModule extends AppModule {
    private static final Logger LOGGER = Logger.getLogger(PreprocessModule.class.getName());
    private int hostDeviceId;

    public PreprocessModule(int id, String name, String appId, int userId, int mips, int ram, long bw, long size, CustomTupleScheduler scheduler, int hostDeviceId) {
        super(id, name, appId, userId, mips, ram, bw, size, "Xen", scheduler, new HashMap<>());
        this.hostDeviceId = hostDeviceId;
        ((CustomTupleScheduler) getCloudletScheduler()).setModule(this);
    }


    protected void processTupleArrival(Tuple tuple) {
        double startTime = CloudSim.clock();
        LOGGER.info("PreprocessModule received tuple at time " + String.format("%.2f", CloudSim.clock()) + ": " + tuple.getTupleType());
        
        if (tuple instanceof DataTuple) {
            DataTuple dt = (DataTuple) tuple;
            Map<String, Object> data = dt.getPayload();
            int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
            int trueFault = ((Number) data.getOrDefault("true_fault", 0)).intValue();
            
            LOGGER.info("Preprocessing sensor data from Machine-" + machineId + 
                       " (true_fault=" + trueFault + ") at time " + String.format("%.2f", CloudSim.clock()));
            
            preprocessData(data, machineId);
            
            double processingLatency = (CloudSim.clock() - startTime) * 1000;
            MetricsCollector.recordPreprocessingLatency(processingLatency);
            
            LOGGER.info("Preprocessing completed for Machine-" + machineId + 
                       " (latency=" + String.format("%.2f", processingLatency) + "ms)");

            forwardToML(data, startTime);
            
        } else {
            LOGGER.warning(" PreprocessModule received non-DataTuple: " + tuple.getClass().getSimpleName());
        }
    }
    
    private void preprocessData(Map<String, Object> data, int machineId) {
        @SuppressWarnings("unchecked")
        List<Double> vibration = (List<Double>) data.get("vibration");
        
        int nanCount = 0;
        double sum = 0, count = 0;
        for (Double v : vibration) {
            if (Double.isNaN(v)) {
                nanCount++;
                continue;
            }
            sum += v;
            count++;
        }
        
        double mean = (count > 0) ? sum / count : 0.0;
        for (int i = 0; i < vibration.size(); i++) {
            if (Double.isNaN(vibration.get(i))) {
                vibration.set(i, mean);
            }
        }
        
        boolean tempMissing = false;
        if (!data.containsKey("temp") || Double.isNaN((Double) data.get("temp"))) {
            data.put("temp", 50.0);
            tempMissing = true;
        }
        
        boolean voltageMissing = false;
        if (!data.containsKey("voltage") || Double.isNaN((Double) data.get("voltage"))) {
            data.put("voltage", 220.0);
            voltageMissing = true;
        }
        
        StringBuilder details = new StringBuilder();
        if (nanCount > 0) details.append("vibration_nans:").append(nanCount).append(" ");
        if (tempMissing) details.append("temp_missing ");
        if (voltageMissing) details.append("voltage_missing ");
        
        if (details.length() > 0) {
            LOGGER.info("Machine-" + machineId + " data cleaning: " + details.toString().trim());
        }
    }
    
    private void forwardToML(Map<String, Object> data, double startTime) {
        int machineId = ((Number) data.getOrDefault("machine_id", 0)).intValue();
        
        DataTuple edgeTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 1000, 1000,
                new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
        edgeTuple.setUserId(getUserId());
        edgeTuple.setPayload(new HashMap<>(data));
        edgeTuple.setTupleType("PROCESSED_TO_EDGE");
        edgeTuple.setDestModuleName("EdgeML");
        sendTuple(edgeTuple, "EdgeML");
        
                    LOGGER.info("Machine-" + machineId + " data forwarded to EdgeML for fast prediction");

        DataTuple cloudTuple = new DataTuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP, 1000, 1, 1000, 1000,
                new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
        cloudTuple.setUserId(getUserId());
        cloudTuple.setPayload(new HashMap<>(data));
        cloudTuple.setTupleType("PROCESSED_TO_CLOUD");
        cloudTuple.setDestModuleName("CloudML");
        sendTuple(cloudTuple, "CloudML");
        
                    LOGGER.info("Machine-" + machineId + " data forwarded to CloudML for comprehensive analysis");
        
        double endToEndLatency = (CloudSim.clock() - startTime) * 1000;
        MetricsCollector.recordEndToEndLatency(endToEndLatency);
    }

    private void sendTuple(DataTuple tuple, String destModule) {
        tuple.setDestModuleName(destModule);
        CloudSim.send(hostDeviceId, hostDeviceId, 0.0, FogEvents.TUPLE_ARRIVAL, tuple);
        LOGGER.info("PreprocessModule sent tuple to " + destModule + " at time " + String.format("%.2f", CloudSim.clock()));
    }
}