package org.fog.test.perfeval;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.fog.application.Application;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.Controller;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementEdgewards;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.distribution.DeterministicDistribution;
import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class IntelliPdM {
    static List<FogDevice> fogDevices = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();
    static int numMachines = 5; 
    static int numEdges = 1; 

    public static void main(String[] args) {
        Log.enable();
        int userId = FogUtils.USER_ID;
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;
        CloudSim.init(userId, calendar, trace_flag);

        try {
            ProcessBuilder pb = new ProcessBuilder("/home/rishiikesh/Rishiikesh/Sem5/Edge Computing/iFogSim/venv/bin/python", "python_ml/train_cnn.py");
            pb.inheritIO();
            pb.start().waitFor();
            ProcessBuilder pbRf = new ProcessBuilder("/home/rishiikesh/Rishiikesh/Sem5/Edge Computing/iFogSim/venv/bin/python", "python_ml/train_rf.py");
            pbRf.inheritIO();
            pbRf.start().waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }

        String appId = "IntelliPdM";
        Application app = createApplication(appId, userId);

        createFogDevices(userId, appId);

        ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
        moduleMapping.addModuleToDevice("Preprocess", "edge-0");
        moduleMapping.addModuleToDevice("EdgeML", "edge-0");
        moduleMapping.addModuleToDevice("CloudML", "cloud");

        Controller controller = new Controller("controller", fogDevices, sensors, actuators);

        controller.submitApplication(app, 0, new ModulePlacementEdgewards(fogDevices, sensors, actuators, app, moduleMapping));

        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        printMetrics();

        SwingUtilities.invokeLater(() -> showGuiMetrics());
    }

    private static void createFogDevices(int userId, String appId) {
        FogDevice cloud = createFogDevice("cloud", 44800, 40000, 100, 10000, 0, 0.01, 16 * 103, 16 * 83.25, 0.01);
        fogDevices.add(cloud);

        for (int i = 0; i < numEdges; i++) {
            FogDevice edge = createFogDevice("edge-" + i, 2800, 4000, 10000, 10000, 1, 0.0, 107.339, 83.4333, 0.01);
            edge.setParentId(cloud.getId());
            cloud.getChildToLatencyMap().put(edge.getId(), 0.01); 
            fogDevices.add(edge);
        }

        for (int i = 0; i < numMachines; i++) {
            Sensor sensor = new CustomSensor("sensor-" + i, "SENSOR", userId, appId, new DeterministicDistribution(5));
            sensor.setGatewayDeviceId(fogDevices.get(1).getId()); 
            sensor.setLatency(1.0);
            fogDevices.get(1).getChildToLatencyMap().put(sensor.getId(), 1.0); 
            sensors.add(sensor);

            Actuator actuator = new Actuator("actuator-" + i, userId, appId, "STOP_ACTUATOR");
            actuator.setGatewayDeviceId(fogDevices.get(1).getId()); 
            actuator.setLatency(1.0);
            fogDevices.get(1).getChildToLatencyMap().put(actuator.getId(), 1.0); 
            actuators.add(actuator);
        }

        System.out.println("Cloud childToLatencyMap: " + cloud.getChildToLatencyMap());
        System.out.println("Edge-0 childToLatencyMap: " + fogDevices.get(1).getChildToLatencyMap());
    }

    private static FogDevice createFogDevice(String nodeName, long mips, int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower, double uplinkLatency) {
        List<Pe> peList = new ArrayList<>();
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));
        int hostId = FogUtils.generateEntityId();
        long storage = 1000000;
        int bw = 10000;
        PowerHost host = new PowerHost(
            hostId,
            new RamProvisionerSimple(ram),
            new BwProvisionerSimple(bw),
            storage,
            peList,
            new StreamOperatorScheduler(peList),
            new FogLinearPowerModel(busyPower, idlePower)
        );

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
            "x86", "Linux", "Xen", host, 10.0, 0.01, 0.01, 0.001, 0.0
        );

        List<Host> hostList = new ArrayList<>();
        hostList.add(host);
        AppModuleAllocationPolicy allocationPolicy = new AppModuleAllocationPolicy(hostList);
        List<Storage> storageList = new ArrayList<>();
        double schedulingInterval = 10.0;

        try {
            FogDevice fogDevice = new FogDevice(nodeName, characteristics, allocationPolicy, storageList, schedulingInterval, upBw, downBw, uplinkLatency, ratePerMips);
            fogDevice.setLevel(level);
            return fogDevice;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static Application createApplication(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);
        application.addAppModule("Preprocess", 1000);
        application.addAppModule("EdgeML", 2000);
        application.addAppModule("CloudML", 4000);

        application.addAppEdge("SENSOR", "Preprocess", 100, 200, "SENSOR", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("Preprocess", "EdgeML", 500, 100, "PROCESSED_DATA", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("Preprocess", "CloudML", 500, 100, "PROCESSED_DATA", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("EdgeML", "STOP_ACTUATOR", 100, 50, "STOP", Tuple.ACTUATOR, AppEdge.ACTUATOR);
        application.addAppEdge("CloudML", "STOP_ACTUATOR", 100, 50, "STOP", Tuple.ACTUATOR, AppEdge.ACTUATOR);
        application.addAppEdge("CloudML", "EdgeML", 1000000, 500000, "UPDATE_MODEL", Tuple.UP, AppEdge.MODULE);

        application.addTupleMapping("Preprocess", "SENSOR", "PROCESSED_DATA", new FractionalSelectivity(1.0));
        application.addTupleMapping("EdgeML", "PROCESSED_DATA", "STOP", new FractionalSelectivity(0.2));
        application.addTupleMapping("CloudML", "PROCESSED_DATA", "STOP", new FractionalSelectivity(0.2));

        List<AppLoop> loops = new ArrayList<>();
        List<String> loop1 = new ArrayList<>();
        loop1.add("SENSOR");
        loop1.add("Preprocess");
        loop1.add("EdgeML");
        loop1.add("STOP_ACTUATOR");
        loops.add(new AppLoop(loop1));
        List<String> loop2 = new ArrayList<>();
        loop2.add("SENSOR");
        loop2.add("Preprocess");
        loop2.add("CloudML");
        loop2.add("STOP_ACTUATOR");
        loops.add(new AppLoop(loop2));
        application.setLoops(loops);

        return application;
    }

    private static void printMetrics() {
        Log.printLine("Simulation time: " + CloudSim.clock());
        for (FogDevice fd : fogDevices) {
            Log.printLine(fd.getName() + " energy: " + fd.getEnergyConsumption());
        }
    }

    private static void showGuiMetrics() {
        JFrame frame = new JFrame("IntelliPdM Metrics");
        frame.setSize(400, 300);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JTextArea textArea = new JTextArea();
        textArea.append("Simulation time: " + CloudSim.clock() + "\n");
        for (FogDevice fd : fogDevices) {
            textArea.append(fd.getName() + " energy: " + fd.getEnergyConsumption() + "\n");
        }
        frame.add(new JScrollPane(textArea));
        frame.setVisible(true);
    }
}