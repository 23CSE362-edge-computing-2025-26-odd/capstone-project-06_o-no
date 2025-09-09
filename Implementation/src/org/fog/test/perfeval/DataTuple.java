package org.fog.test.perfeval;

import org.cloudbus.cloudsim.UtilizationModel;
import org.fog.entities.Tuple;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Custom Tuple class to include payload for sensor data.
 */
public class DataTuple extends Tuple {
    private Map<String, Object> payload;
    private List<String> destModuleNames;

    public DataTuple(String appId, int cloudletId, int direction, long cloudletLength, int pesNumber,
                     long nwLength, long outputSize, UtilizationModel cpuUtilizationModel,
                     UtilizationModel ramUtilizationModel, UtilizationModel bwUtilizationModel) {
        super(appId, cloudletId, direction, cloudletLength, pesNumber, nwLength, outputSize,
              cpuUtilizationModel, ramUtilizationModel, bwUtilizationModel);
        this.destModuleNames = new ArrayList<>();
    }

    public Map<String, Object> getPayload() {
        return payload;
    }

    public void setPayload(Map<String, Object> payload) {
        this.payload = payload;
    }

    public List<String> getDestModuleNames() {
        return destModuleNames;
    }

    public void setDestModuleNames(List<String> destModuleNames) {
        this.destModuleNames = destModuleNames;
        if (!destModuleNames.isEmpty()) {
            // Set the first module as the primary destination for compatibility with parent Tuple class
            setDestModuleName(destModuleNames.get(0));
        }
    }
}