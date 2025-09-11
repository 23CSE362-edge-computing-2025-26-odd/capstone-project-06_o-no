#!/usr/bin/env python3
"""
Configuration Reader for IntelliPdM Simulations
Reads settings from config.properties file
"""

import os
import configparser

class ConfigReader:
    def __init__(self, config_file="config.properties"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from properties file"""
        if not os.path.exists(self.config_file):
            print(f"WARNING: Configuration file '{self.config_file}' not found, using defaults")
            self.set_defaults()
            return
        
        try:
            # Read Java properties file
            with open(self.config_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    self.config[key.strip()] = value.strip()
            
            print(f"Configuration loaded from {self.config_file}")
            self.validate_config()
            
        except Exception as e:
            print(f"Error reading config file: {e}")
            print("   Using default configuration...")
            self.set_defaults()
    
    def set_defaults(self):
        """Set default configuration values"""
        self.config = {
            'simDuration': '100.0',
            'numMachines': '5',
            'initialNumEdges': '1',
            'loadThreshold': '0.8',
            'monitorInterval': '10.0',
            'pythonExec': 'python',
            'projectDir': os.getcwd(),
            'debug': 'false',
            'logLevel': 'INFO'
        }
    
    def validate_config(self):
        """Validate and fix configuration values"""
        # Ensure projectDir exists
        project_dir = self.config.get('projectDir', os.getcwd())
        if not os.path.exists(project_dir):
            print(f"WARNING: Project directory '{project_dir}' not found, using current directory")
            self.config['projectDir'] = os.getcwd()
        
        # Validate python executable
        python_exec = self.config.get('pythonExec', 'python')
        try:
            import subprocess
            result = subprocess.run([python_exec, '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"WARNING: Python executable '{python_exec}' not working, trying 'python3'")
                self.config['pythonExec'] = 'python3'
        except:
            self.config['pythonExec'] = 'python3'
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_int(self, key, default=0):
        """Get configuration value as integer"""
        try:
            return int(float(self.config.get(key, default)))
        except:
            return default
    
    def get_float(self, key, default=0.0):
        """Get configuration value as float"""
        try:
            return float(self.config.get(key, default))
        except:
            return default
    
    def get_bool(self, key, default=False):
        """Get configuration value as boolean"""
        value = self.config.get(key, str(default)).lower()
        return value in ['true', '1', 'yes', 'on']
    
    # Convenience methods for specific configuration values
    def get_num_machines(self):
        """Get number of machines"""
        return self.get_int('numMachines', 3)
    
    def get_sim_duration(self):
        """Get simulation duration in seconds"""
        return self.get_float('simDuration', 15.0)
    
    def get_monitor_interval(self):
        """Get monitoring interval in seconds"""
        return self.get_float('monitorInterval', 3.0)
    
    def get_edge_threshold(self):
        """Get edge confidence threshold"""
        return self.get_float('loadThreshold', 0.7)
    
    def get_python_executable(self):
        """Get Python executable path"""
        return self.config.get('pythonExec', 'python')
    
    def display_config(self):
        """Display current configuration"""
        print("📋 Current Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        print()

# Global configuration instance
config = ConfigReader()
