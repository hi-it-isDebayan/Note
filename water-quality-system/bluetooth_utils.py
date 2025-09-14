# Bluetooth Utilities
import time
import pandas as pd
import numpy as np
from datetime import datetime

class BluetoothManager:
    """Class to manage Bluetooth connections to water quality sensors"""
    
    def _init_(self):
        self.connected_devices = {}
        self.available_devices = []
        self.sensor_types = {
            "pH": {"min": 6.0, "max": 8.5, "unit": "pH"},
            "turbidity": {"min": 0.1, "max": 50.0, "unit": "NTU"},
            "TDS": {"min": 50, "max": 1000, "unit": "ppm"},
            "chlorine": {"min": 0.1, "max": 2.0, "unit": "mg/L"},
            "temperature": {"min": 15, "max": 35, "unit": "°C"}
        }
    
    def scan_devices(self):
        """Scan for available Bluetooth devices (simulated for now)"""
        # Simulate scanning for devices
        devices = [
            {"name": "pH_Sensor_001", "address": "00:1A:7D:DA:71:13", "signal_strength": -65, "type": "pH"},
            {"name": "Turbidity_Sensor_002", "address": "00:1A:7D:DA:71:14", "signal_strength": -72, "type": "turbidity"},
            {"name": "TDS_Sensor_003", "address": "00:1A:7D:DA:71:15", "signal_strength": -58, "type": "TDS"},
            {"name": "Chlorine_Sensor_004", "address": "00:1A:7D:DA:71:16", "signal_strength": -62, "type": "chlorine"},
            {"name": "Temperature_Sensor_005", "address": "00:1A:7D:DA:71:17", "signal_strength": -68, "type": "temperature"},
        ]
        self.available_devices = devices
        return devices
    
    def connect_device(self, device_address):
        """Connect to a specific device (simulated for now)"""
        # Simulate connection process
        time.sleep(1)
        
        # Find the device
        device = next((d for d in self.available_devices if d["address"] == device_address), None)
        if device:
            # Simulate successful connection
            self.connected_devices[device_address] = {
                "name": device["name"],
                "type": device["type"],
                "connected": True,
                "battery_level": np.random.randint(60, 100),
                "last_connection": time.time(),
                "condition": np.random.choice(["Excellent", "Good", "Fair", "Needs Maintenance"], 
                                            p=[0.3, 0.4, 0.2, 0.1])
            }
            return True
        return False
    
    def disconnect_device(self, device_address):
        """Disconnect from a device"""
        if device_address in self.connected_devices:
            del self.connected_devices[device_address]
            return True
        return False
    
    def get_device_status(self, device_address):
        """Get the status of a connected device"""
        return self.connected_devices.get(device_address, {})
    
    def read_data(self, device_address):
        """Read data from a connected device (simulated for now)"""
        if device_address not in self.connected_devices:
            return None
        
        # Get device info
        device_info = self.connected_devices[device_address]
        sensor_type = device_info["type"]
        
        # Get sensor range
        sensor_range = self.sensor_types.get(sensor_type, {"min": 0, "max": 1})
        
        # Simulate reading based on sensor type
        if sensor_type == "pH":
            value = round(np.random.uniform(sensor_range["min"], sensor_range["max"]), 2)
            return {"pH": value}
        elif sensor_type == "turbidity":
            value = round(np.random.uniform(sensor_range["min"], sensor_range["max"]), 2)
            return {"turbidity": value}
        elif sensor_type == "TDS":
            value = round(np.random.uniform(sensor_range["min"], sensor_range["max"]), 1)
            return {"TDS": value}
        elif sensor_type == "chlorine":
            value = round(np.random.uniform(sensor_range["min"], sensor_range["max"]), 2)
            return {"residual_chlorine": value}
        elif sensor_type == "temperature":
            value = round(np.random.uniform(sensor_range["min"], sensor_range["max"]), 1)
            return {"temperature": value}
        
        return None


class SensorDataCollector:
    """Class to collect and manage data from multiple sensors"""
    
    def _init_(self):
        self.sensor_data = pd.DataFrame()
        self.bluetooth_manager = BluetoothManager()
    
    def collect_all_data(self):
        """Collect data from all connected sensors"""
        readings = {}
        timestamp = datetime.now()
        
        for address in self.bluetooth_manager.connected_devices:
            data = self.bluetooth_manager.read_data(address)
            if data:
                readings.update(data)
        
        if readings:
            readings["timestamp"] = timestamp
            new_data = pd.DataFrame([readings])
            
            if self.sensor_data.empty:
                self.sensor_data = new_data
            else:
                self.sensor_data = pd.concat([self.sensor_data, new_data], ignore_index=True)
            
            return new_data
        
        return None
    
    def get_latest_readings(self):
        """Get the latest readings from all sensors"""
        if self.sensor_data.empty:
            return None
        
        return self.sensor_data.iloc[-1].to_dict()
    
    def get_historical_data(self, hours=24):
        """Get historical data for a specific time period"""
        if self.sensor_data.empty:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        return self.sensor_data[self.sensor_data["timestamp"] >= cutoff_time]
    
    def calculate_risk_score(self, readings):
        """Calculate a risk score based on sensor readings"""
        if not readings:
            return 0
        
        risk_score = 0
        
        # pH component (ideal is 7)
        if 'pH' in readings:
            risk_score += min(100, abs(readings['pH'] - 7) / 0.5 * 25)
        
        # Turbidity component (lower is better)
        if 'turbidity' in readings:
            risk_score += min(100, readings['turbidity'] / 5 * 25)
        
        # TDS component (higher is worse)
        if 'TDS' in readings:
            risk_score += min(100, readings['TDS'] / 1000 * 25)
        
        # Chlorine component (both too low and too high are bad)
        if 'residual_chlorine' in readings:
            chlorine = readings['residual_chlorine']
            if chlorine < 0.2:
                risk_score += (0.2 - chlorine) / 0.2 * 25
            elif chlorine > 2.0:
                risk_score += (chlorine - 2.0) / 2.0 * 25
        
        return min(100, risk_score)