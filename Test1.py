"""
TrueBlock - A TrueCaller Clone with Audio Deepfake Detection Created using Python 3.12
This application provides:
1. Caller identification (similar to TrueCaller)
2. CallGuard AI feature to detect AI-generated or pre-recorded voices
"""

import os
import sys
import time
import sqlite3
import hashlib
import json
import threading
import queue
import logging
from datetime import datetime
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QLineEdit,
                            QListWidget, QListWidgetItem, QTabWidget, QDialog,
                            QFileDialog, QMessageBox, QProgressBar, QComboBox)
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import pickle
import phonenumbers
from phonenumbers import geocoder, carrier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trueblock.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
RECORDING_DURATION = 5  # seconds
FEATURE_EXTRACTION_WINDOW = 1  # second

# Database path
DB_PATH = "trueblock.db"

class CallerDatabase:
    """Manages the database for caller information"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS contacts (
                phone_number TEXT PRIMARY KEY,
                name TEXT,
                is_spam BOOLEAN,
                country TEXT,
                carrier TEXT,
                added_on TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT,
                timestamp TIMESTAMP,
                duration INTEGER,
                call_type TEXT,
                deepfake_probability REAL,
                recording_probability REAL,
                FOREIGN KEY (phone_number) REFERENCES contacts(phone_number)
            )
            ''')
            
            # Add some sample spam numbers
            sample_spam = [
                ("+919876543210", "Suspected Spam", True, "IN", "Unknown", datetime.now()),
                ("+919876543211", "Telemarketing", True, "IN", "Unknown", datetime.now()),
                ("+919876543212", "Scam Likely", True, "IN", "Unknown", datetime.now())
            ]
            
            cursor.executemany('''
            INSERT OR IGNORE INTO contacts VALUES (?, ?, ?, ?, ?, ?)
            ''', sample_spam)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    def add_contact(self, phone_number, name, is_spam=False):
        """Add a new contact to the database"""
        try:
            # Parse phone number to get country and carrier info
            parsed_number = phonenumbers.parse(phone_number)
            country = geocoder.description_for_number(parsed_number, "en")
            carrier_name = carrier.name_for_number(parsed_number, "en")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO contacts VALUES (?, ?, ?, ?, ?, ?)
            ''', (phone_number, name, is_spam, country, carrier_name, datetime.now()))
            
            conn.commit()
            conn.close()
            logger.info(f"Added contact: {name} ({phone_number})")
            return True
        
        except Exception as e:
            logger.error(f"Error adding contact: {str(e)}")
            return False
    
    def get_contact_info(self, phone_number):
        """Get information about a contact by phone number"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM contacts WHERE phone_number = ?
            ''', (phone_number,))
            
            contact = cursor.fetchone()
            conn.close()
            
            if contact:
                return {
                    "phone_number": contact[0],
                    "name": contact[1],
                    "is_spam": bool(contact[2]),
                    "country": contact[3],
                    "carrier": contact[4],
                    "added_on": contact[5]
                }
            else:
                # Try to get basic information from the phone number
                try:
                    parsed_number = phonenumbers.parse(phone_number)
                    country = geocoder.description_for_number(parsed_number, "en")
                    carrier_name = carrier.name_for_number(parsed_number, "en")
                    
                    return {
                        "phone_number": phone_number,
                        "name": "Unknown",
                        "is_spam": False,
                        "country": country,
                        "carrier": carrier_name,
                        "added_on": None
                    }
                except:
                    return None
        
        except sqlite3.Error as e:
            logger.error(f"Error getting contact: {str(e)}")
            return None
    
    def log_call(self, phone_number, duration, call_type, deepfake_prob=0.0, recording_prob=0.0):
        """Log a call to the call history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the contact exists, add as unknown if not
            cursor.execute("SELECT COUNT(*) FROM contacts WHERE phone_number = ?", (phone_number,))
            if cursor.fetchone()[0] == 0:
                self.add_contact(phone_number, "Unknown")
            
            cursor.execute('''
            INSERT INTO call_history 
            (phone_number, timestamp, duration, call_type, deepfake_probability, recording_probability)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (phone_number, datetime.now(), duration, call_type, deepfake_prob, recording_prob))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged call: {phone_number}, type: {call_type}")
            return True
        
        except Exception as e:
            logger.error(f"Error logging call: {str(e)}")
            return False
    
    def get_call_history(self, limit=50):
        """Get recent call history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT ch.id, ch.phone_number, c.name, ch.timestamp, ch.duration, 
                   ch.call_type, ch.deepfake_probability, ch.recording_probability
            FROM call_history ch
            LEFT JOIN contacts c ON ch.phone_number = c.phone_number
            ORDER BY ch.timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            history = cursor.fetchall()
            conn.close()
            
            result = []
            for call in history:
                result.append({
                    "id": call[0],
                    "phone_number": call[1],
                    "name": call[2] if call[2] else "Unknown",
                    "timestamp": call[3],
                    "duration": call[4],
                    "call_type": call[5],
                    "deepfake_probability": call[6],
                    "recording_probability": call[7]
                })
            
            return result
        
        except sqlite3.Error as e:
            logger.error(f"Error getting call history: {str(e)}")
            return []
    
    def search_contacts(self, query):
        """Search for contacts by name or number"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use LIKE for partial matches
            cursor.execute('''
            SELECT * FROM contacts
            WHERE name LIKE ? OR phone_number LIKE ?
            ORDER BY name
            ''', (f'%{query}%', f'%{query}%'))
            
            contacts = cursor.fetchall()
            conn.close()
            
            result = []
            for contact in contacts:
                result.append({
                    "phone_number": contact[0],
                    "name": contact[1],
                    "is_spam": bool(contact[2]),
                    "country": contact[3],
                    "carrier": contact[4],
                    "added_on": contact[5]
                })
            
            return result
        
        except sqlite3.Error as e:
            logger.error(f"Error searching contacts: {str(e)}")
            return []

class AudioFeatureExtractor:
    """Extract audio features for deepfake and recording detection"""
    
    def __init__(self):
        pass
    
    def extract_features(self, audio_data, sample_rate):
        """Extract audio features from the given audio data"""
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract features
            features = {}
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            features['mel_mean'] = np.mean(mel_spec)
            features['mel_std'] = np.std(mel_spec)
            
            # Temporal features
            features['rms_energy'] = np.sqrt(np.mean(np.square(audio_data)))
            
            # Convert to numpy array in consistent order for model input
            feature_names = sorted(features.keys())
            feature_vector = np.array([features[name] for name in feature_names])
            
            return feature_vector
        
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            # Return a zero vector with expected size (40 features)
            return np.zeros(40)

class DeepfakeDetectionModel:
    """Model for detecting AI-generated voices (deepfakes)"""
    
    def __init__(self, model_path=None):
        """Initialize the deepfake detection model"""
        self.model = None
        self.feature_extractor = AudioFeatureExtractor()
        
        # Try to load model if exists, otherwise train a new one
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded deepfake detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self._train_default_model()
        else:
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a basic detection model with default parameters"""
        logger.info("Training default deepfake detection model")
        
        # Create a simple random forest classifier with reasonable defaults
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Since we don't have real data, we'll create a simple model based on 
        # engineered features that simulates what we might expect
        # In a real application, you would train this with actual labeled data
        
        # This is just a placeholder model
        X = np.random.random((100, 40))  # 40 features
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        self.model.fit(X, y)
        logger.info("Default model training complete")
    
    def predict(self, audio_data, sample_rate):
        """Predict if the audio is a deepfake"""
        if self.model is None:
            logger.error("Model not initialized")
            return 0.5  # Return uncertainty
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_data, sample_rate)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([features])[0][1]
                return proba
            else:
                # If model doesn't support probabilities, return binary prediction
                pred = self.model.predict([features])[0]
                return float(pred)
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0.5  # Return uncertainty

class RecordingDetectionModel:
    """Model for detecting if audio is pre-recorded or live"""
    
    def __init__(self, model_path=None):
        """Initialize the recording detection model"""
        self.model = None
        self.feature_extractor = AudioFeatureExtractor()
        
        # Try to load model if exists, otherwise train a new one
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded recording detection model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self._train_default_model()
        else:
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a basic recording detection model with default parameters"""
        logger.info("Training default recording detection model")
        
        # Create a simple random forest classifier with reasonable defaults
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Since we don't have real data, we'll create a simple model based on 
        # engineered features that simulates what we might expect
        # In a real application, you would train this with actual labeled data
        
        # This is just a placeholder model
        X = np.random.random((100, 40))  # 40 features
        y = np.random.randint(0, 2, 100)  # Binary classification
        
        self.model.fit(X, y)
        logger.info("Default model training complete")
    
    def predict(self, audio_data, sample_rate):
        """Predict if the audio is pre-recorded"""
        if self.model is None:
            logger.error("Model not initialized")
            return 0.5  # Return uncertainty
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_data, sample_rate)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([features])[0][1]
                return proba
            else:
                # If model doesn't support probabilities, return binary prediction
                pred = self.model.predict([features])[0]
                return float(pred)
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0.5  # Return uncertainty

class CallGuardAI:
    """Main class for CallGuard AI functionality"""
    
    def __init__(self):
        """Initialize CallGuard AI components"""
        self.deepfake_model = DeepfakeDetectionModel()
        self.recording_model = RecordingDetectionModel()
        self.sample_rate = SAMPLE_RATE
        self.recording = False
        self.audio_buffer = queue.Queue()
        self.recording_thread = None
        self.analysis_thread = None
        self.processing_active = False
        
        # Stats for the current call
        self.current_stats = {
            "deepfake_probabilities": [],
            "recording_probabilities": []
        }
    
    def start_recording(self):
        """Start recording audio for analysis"""
        if self.recording:
            return False
        
        self.recording = True
        self.processing_active = True
        self.current_stats = {
            "deepfake_probabilities": [],
            "recording_probabilities": []
        }
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analyze_audio)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        logger.info("CallGuard AI recording started")
        return True
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.recording:
            return False
        
        self.recording = False
        
        # Wait for threads to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        self.processing_active = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        logger.info("CallGuard AI recording stopped")
        
        # Calculate final stats
        final_stats = self.get_final_stats()
        return final_stats
    
    def _record_audio(self):
        """Record audio continuously and add to buffer"""
        try:
            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio recording status: {status}")
                self.audio_buffer.put(indata.copy())
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
                while self.recording:
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            self.recording = False
    
    def _analyze_audio(self):
        """Analyze audio for deepfake and recording detection"""
        try:
            window_size = int(FEATURE_EXTRACTION_WINDOW * self.sample_rate)
            audio_data = np.zeros(window_size)
            buffer_index = 0
            
            while self.processing_active:
                try:
                    # Get data from buffer with timeout
                    chunk = self.audio_buffer.get(timeout=1.0)
                    
                    # Add to our analysis window
                    chunk_size = len(chunk)
                    if buffer_index + chunk_size >= window_size:
                        # Window is full, analyze it
                        remaining = window_size - buffer_index
                        audio_data[buffer_index:] = chunk[:remaining, 0]
                        
                        # Analyze full window
                        self._analyze_window(audio_data)
                        
                        # Start new window with remaining data
                        audio_data = np.zeros(window_size)
                        buffer_index = 0
                        
                        # Add remaining data to new window
                        overflow = chunk_size - remaining
                        if overflow > 0:
                            audio_data[:overflow] = chunk[remaining:, 0]
                            buffer_index = overflow
                    else:
                        # Add chunk to window
                        audio_data[buffer_index:buffer_index + chunk_size] = chunk[:, 0]
                        buffer_index += chunk_size
                
                except queue.Empty:
                    # Timeout, continue
                    pass
        
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
    
    def _analyze_window(self, audio_data):
        """Analyze a window of audio data"""
        try:
            # Run deepfake detection
            deepfake_prob = self.deepfake_model.predict(audio_data, self.sample_rate)
            
            # Run recording detection
            recording_prob = self.recording_model.predict(audio_data, self.sample_rate)
            
            # Store results
            self.current_stats["deepfake_probabilities"].append(deepfake_prob)
            self.current_stats["recording_probabilities"].append(recording_prob)
            
            logger.debug(f"Analysis window - Deepfake: {deepfake_prob:.2f}, Recording: {recording_prob:.2f}")
        
        except Exception as e:
            logger.error(f"Error in window analysis: {str(e)}")
    
    def get_current_stats(self):
        """Get current statistics about the call"""
        if not self.current_stats["deepfake_probabilities"]:
            return {
                "deepfake_probability": 0.0,
                "recording_probability": 0.0,
                "is_deepfake": False,
                "is_recording": False,
                "confidence": 0.0
            }
        
        deepfake_prob = np.mean(self.current_stats["deepfake_probabilities"])
        recording_prob = np.mean(self.current_stats["recording_probabilities"])
        
        return {
            "deepfake_probability": deepfake_prob,
            "recording_probability": recording_prob,
            "is_deepfake": deepfake_prob > 0.7,  # Threshold
            "is_recording": recording_prob > 0.7,  # Threshold
            "confidence": min(len(self.current_stats["deepfake_probabilities"]) / 10.0, 1.0)
        }
    
    def get_final_stats(self):
        """Get final statistics after call is complete"""
        stats = self.get_current_stats()
        
        # Add extra details for final stats
        if len(self.current_stats["deepfake_probabilities"]) > 0:
            stats["deepfake_std"] = np.std(self.current_stats["deepfake_probabilities"])
            stats["recording_std"] = np.std(self.current_stats["recording_probabilities"])
            stats["analysis_windows"] = len(self.current_stats["deepfake_probabilities"])
        
        return stats

class TrueBlockMainWindow(QMainWindow):
    """Main application window for TrueBlock"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.db = CallerDatabase()
        self.callguard = CallGuardAI()
        
        # Call simulation variables
        self.current_call = None
        self.call_start_time = None
        self.call_timer = QTimer()
        self.call_timer.timeout.connect(self.update_call_duration)
        
        # UI setup
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("TrueBlock - TrueCaller Clone with CallGuard AI")
        self.setGeometry(100, 100, 800, 600)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.dialer_tab = QWidget()
        self.history_tab = QWidget()
        self.contacts_tab = QWidget()
        self.settings_tab = QWidget()
        
        # Add tabs
        self.tabs.addTab(self.dialer_tab, "Dialer")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.contacts_tab, "Contacts")
        self.tabs.addTab(self.settings_tab, "Settings")
        
        # Set up tab contents
        self.setup_dialer_tab()
        self.setup_history_tab()
        self.setup_contacts_tab()
        self.setup_settings_tab()
        
        # Load call history
        self.load_call_history()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def setup_dialer_tab(self):
        """Set up the dialer tab"""
        layout = QVBoxLayout()
        
        # Phone number input
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Enter phone number")
        self.phone_input.setFont(QFont("Arial", 16))
        layout.addWidget(self.phone_input)
        
        # Call button
        call_button_layout = QHBoxLayout()
        
        self.call_button = QPushButton("Call")
        self.call_button.setFont(QFont("Arial", 14))
        self.call_button.clicked.connect(self.start_call)
        call_button_layout.addWidget(self.call_button)
        
        self.hangup_button = QPushButton("Hang Up")
        self.hangup_button.setFont(QFont("Arial", 14))
        self.hangup_button.clicked.connect(self.end_call)
        self.hangup_button.setEnabled(False)
        call_button_layout.addWidget(self.hangup_button)
        
        layout.addLayout(call_button_layout)
        
        # Caller info
        self.caller_info_widget = QWidget()
        caller_info_layout = QVBoxLayout()
        
        self.caller_name_label = QLabel("No Active Call")
        self.caller_name_label.setFont(QFont("Arial", 18, QFont.Bold))
        caller_info_layout.addWidget(self.caller_name_label)
        
        self.caller_number_label = QLabel("")
        self.caller_number_label.setFont(QFont("Arial", 14))
        caller_info_layout.addWidget(self.caller_number_label)
        
        self.caller_location_label = QLabel("")
        self.caller_location_label.setFont(QFont("Arial", 12))
        caller_info_layout.addWidget(self.caller_location_label)
        
        self.call_duration_label = QLabel("")
        self.call_duration_label.setFont(QFont("Arial", 12))
        caller_info_layout.addWidget(self.call_duration_label)
        
        self.caller_info_widget.setLayout(caller_info_layout)
        layout.addWidget(self.caller_info_widget)
        
        # CallGuard status
        self.callguard_status_group = QWidget()
        callguard_layout = QVBoxLayout()
        
        self.callguard_title = QLabel("CallGuard AI Status")
        self.callguard_title.setFont(QFont("Arial", 14, QFont.Bold))
        callguard_layout.addWidget(self.callguard_title)
        
        self.deepfake_label = QLabel("Deepfake Detection: Not Active")
        callguard_layout.addWidget(self.deepfake_label)
        
        self.deepfake_bar = QProgressBar()
        self.deepfake_bar.setRange(0, 100)
        self.deepfake_bar.setValue(0)
        callguard_layout.addWidget(self.deepfake_bar)
        
        self.recording_label = QLabel("Recording Detection: Not Active")
        callguard_layout.addWidget(self.recording_label)
        
        self.recording_bar = QProgressBar()
        self.recording_bar.setRange(0, 100)
        self.recording_bar.setValue(0)
        callguard_layout.addWidget(self.recording_bar)
        
        self.confidence_label = QLabel("Analysis Confidence: 0%")
        callguard_layout.addWidget(self.confidence_label)
        
        self.callguard_status_group.setLayout(callguard_layout)
        layout.addWidget(self.callguard_status_group)
        self.callguard_status_group.setVisible(False)
        
        # Add a spacer
        layout.addStretch()
        
        # Call simulation buttons (for demo purposes)
        sim_layout = QHBoxLayout()
        
        self.sim_normal_button = QPushButton("Simulate Normal Call")
        self.sim_normal_button.clicked.connect(self.simulate_normal_call)
        sim_layout.addWidget(self.sim_normal_button)
        
        self.sim_deepfake_button = QPushButton("Simulate Deepfake")
        self.sim_deepfake_button.clicked.connect(self.simulate_deepfake_call)
        sim_layout.addWidget(self.sim_deepfake_button)
        
        self.sim_recording_button = QPushButton("Simulate Recording")
        self.sim_recording_button.clicked.connect(self.simulate_recording_call)
        sim_layout.addWidget(self.sim_recording_button)
        
        layout.addLayout(sim_layout)
        
        # Setup timer for CallGuard updates
        self.callguard_timer = QTimer()
        self.callguard_timer.timeout.connect(self.update_callguard_status)
        
        self.dialer_tab.setLayout(layout)
    
    def setup_history_tab(self):
        """Set up the call history tab"""
        layout = QVBoxLayout()
        
        # History list
        self.history_list = QListWidget()
        self.history_list.setFont(QFont("Arial", 12))
        layout.addWidget(self.history_list)
        
        # Refresh button
        refresh_button = QPushButton("Refresh History")
        refresh_button.clicked.connect(self.load_call_history)
        layout.addWidget(refresh_button)
        
        self.history_tab.setLayout(layout)
    
    def setup_contacts_tab(self):
        """Set up the contacts tab"""
        layout = QVBoxLayout()
        
        # Search input
        search_layout = QHBoxLayout()
        
        self.contact_search = QLineEdit()
        self.contact_search.setPlaceholderText("Search contacts...")
        self.contact_search.textChanged.connect(self.search_contacts)
        search_layout.addWidget(self.contact_search)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(lambda: self.search_contacts(self.contact_search.text()))
        search_layout.addWidget(search_button)
        
        layout.addLayout(search_layout)
        
        # Contacts list
        self.contacts_list = QListWidget()
        self.contacts_list.setFont(QFont("Arial", 12))
        layout.addWidget(self.contacts_list)
        
        # Add contact button
        add_button = QPushButton("Add New Contact")
        add_button.clicked.connect(self.show_add_contact_dialog)
        layout.addWidget(add_button)
        
        self.contacts_tab.setLayout(layout)
        
        # Initial contacts load
        self.search_contacts("")
    
    def setup_settings_tab(self):
        """Set up the settings tab"""
        layout = QVBoxLayout()
        
        # CallGuard settings
        group_label = QLabel("CallGuard AI Settings")
        group_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(group_label)
        
        # Deepfake threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Deepfake Alert Threshold:"))
        
        self.deepfake_threshold = QComboBox()
        for i in range(5, 10):
            self.deepfake_threshold.addItem(f"{i/10:.1f}", i/10)
        self.deepfake_threshold.setCurrentIndex(2)  # Default 0.7
        threshold_layout.addWidget(self.deepfake_threshold)
        
        layout.addLayout(threshold_layout)
        
        # Recording threshold
        rec_threshold_layout = QHBoxLayout()
        rec_threshold_layout.addWidget(QLabel("Recording Alert Threshold:"))
        
        self.recording_threshold = QComboBox()
        for i in range(5, 10):
            self.recording_threshold.addItem(f"{i/10:.1f}", i/10)
        self.recording_threshold.setCurrentIndex(2)  # Default 0.7
        rec_threshold_layout.addWidget(self.recording_threshold)
        
        layout.addLayout(rec_threshold_layout)
        
        # Auto-reject options
        self.auto_reject_deepfakes = QComboBox()
        self.auto_reject_deepfakes.addItem("Do not auto-reject deepfakes")
        self.auto_reject_deepfakes.addItem("Ask before rejecting deepfakes")
        self.auto_reject_deepfakes.addItem("Auto-reject all deepfakes")
        layout.addWidget(self.auto_reject_deepfakes)
        
        # Reset models button
        reset_button = QPushButton("Reset Detection Models")
        reset_button.clicked.connect(self.reset_models)
        layout.addWidget(reset_button)
        
        # Add a spacer
        layout.addStretch()
        
        # About section
        about_label = QLabel("TrueBlock v1.0 - A TrueCaller Clone with CallGuard AI")
        about_label.setFont(QFont("Arial", 10))
        layout.addWidget(about_label)
        
        self.settings_tab.setLayout(layout)
    
    def start_call(self):
        """Start a call with the entered phone number"""
        phone_number = self.phone_input.text().strip()
        if not phone_number:
            QMessageBox.warning(self, "Error", "Please enter a phone number")
            return
        
        # Get contact info
        contact_info = self.db.get_contact_info(phone_number)
        
        # Start call
        self.current_call = phone_number
        self.call_start_time = time.time()
        
        # Update UI
        self.caller_name_label.setText(contact_info["name"] if contact_info else "Unknown")
        self.caller_number_label.setText(phone_number)
        self.caller_location_label.setText(f"{contact_info['country']} • {contact_info['carrier']}" if contact_info else "")
        self.call_duration_label.setText("Call duration: 00:00")
        
        # Update buttons
        self.call_button.setEnabled(False)
        self.hangup_button.setEnabled(True)
        
        # Start call timer
        self.call_timer.start(1000)
        
        # Start CallGuard
        self.callguard.start_recording()
        self.callguard_status_group.setVisible(True)
        self.callguard_timer.start(500)  # Update every 500ms
        
        self.statusBar().showMessage(f"Call started with {phone_number}")
    
    def end_call(self):
        """End the current call"""
        if not self.current_call:
            return
        
        # Stop timers
        self.call_timer.stop()
        self.callguard_timer.stop()
        
        # Calculate duration
        duration = int(time.time() - self.call_start_time)
        
        # Stop CallGuard and get results
        final_stats = self.callguard.stop_recording()
        
        # Log call
        self.db.log_call(
            self.current_call, 
            duration, 
            "outgoing", 
            final_stats["deepfake_probability"],
            final_stats["recording_probability"]
        )
        
        # Reset UI
        self.caller_name_label.setText("No Active Call")
        self.caller_number_label.setText("")
        self.caller_location_label.setText("")
        self.call_duration_label.setText("")
        self.callguard_status_group.setVisible(False)
        
        # Reset buttons
        self.call_button.setEnabled(True)
        self.hangup_button.setEnabled(False)
        
        # Reset call
        self.current_call = None
        self.call_start_time = None
        
        # Show summary
        if final_stats["is_deepfake"] or final_stats["is_recording"]:
            warning_msg = "Call Analysis:\n\n"
            
            if final_stats["is_deepfake"]:
                warning_msg += f"⚠️ AI-GENERATED VOICE DETECTED ({final_stats['deepfake_probability']:.1%} probability)\n"
            
            if final_stats["is_recording"]:
                warning_msg += f"⚠️ PRE-RECORDED AUDIO DETECTED ({final_stats['recording_probability']:.1%} probability)\n"
            
            warning_msg += "\nThis call may have been a scam attempt."
            
            QMessageBox.warning(self, "CallGuard AI Warning", warning_msg)
        
        # Refresh history
        self.load_call_history()
        
        self.statusBar().showMessage("Call ended")
    
    def update_call_duration(self):
        """Update the call duration display"""
        if not self.call_start_time:
            return
        
        duration = int(time.time() - self.call_start_time)
        minutes = duration // 60
        seconds = duration % 60
        self.call_duration_label.setText(f"Call duration: {minutes:02d}:{seconds:02d}")
    
    def update_callguard_status(self):
        """Update the CallGuard AI status display"""
        if not self.current_call:
            return
        
        stats = self.callguard.get_current_stats()
        
        # Update labels and progress bars
        deepfake_pct = int(stats["deepfake_probability"] * 100)
        recording_pct = int(stats["recording_probability"] * 100)
        confidence_pct = int(stats["confidence"] * 100)
        
        self.deepfake_label.setText(f"Deepfake Detection: {deepfake_pct}%")
        self.deepfake_bar.setValue(deepfake_pct)
        
        # Set color based on value
        if deepfake_pct > 70:
            self.deepfake_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif deepfake_pct > 40:
            self.deepfake_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.deepfake_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        
        self.recording_label.setText(f"Recording Detection: {recording_pct}%")
        self.recording_bar.setValue(recording_pct)
        
        # Set color based on value
        if recording_pct > 70:
            self.recording_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif recording_pct > 40:
            self.recording_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.recording_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        
        self.confidence_label.setText(f"Analysis Confidence: {confidence_pct}%")
        
        # Automatic warnings
        if stats["confidence"] > 0.5:
            if stats["is_deepfake"] and not hasattr(self, '_deepfake_warned'):
                self.statusBar().showMessage("⚠️ WARNING: AI-generated voice detected!")
                self._deepfake_warned = True
            
            if stats["is_recording"] and not hasattr(self, '_recording_warned'):
                self.statusBar().showMessage("⚠️ WARNING: Pre-recorded audio detected!")
                self._recording_warned = True
    
    def load_call_history(self):
        """Load call history into the history tab"""
        self.history_list.clear()
        
        history = self.db.get_call_history()
        for call in history:
            # Format timestamp
            timestamp = datetime.fromisoformat(call["timestamp"]) 
            date_str = timestamp.strftime("%Y-%m-%d %H:%M")
            
            # Format duration
            minutes = call["duration"] // 60
            seconds = call["duration"] % 60
            duration_str = f"{minutes:02d}:{seconds:02d}"
            
            # Format item text
            item_text = f"{call['name']} ({call['phone_number']}) - {date_str} - {duration_str}"
            
            if call["deepfake_probability"] > 0.7 or call["recording_probability"] > 0.7:
                item_text += " ⚠️"
            
            item = QListWidgetItem(item_text)
            
            # Color coding based on spam status
            contact_info = self.db.get_contact_info(call["phone_number"])
            if contact_info and contact_info["is_spam"]:
                item.setForeground(QColor("red"))
            
            self.history_list.addItem(item)
    
    def search_contacts(self, query):
        """Search contacts and display results"""
        self.contacts_list.clear()
        
        contacts = self.db.search_contacts(query)
        for contact in contacts:
            item_text = f"{contact['name']} ({contact['phone_number']})"
            
            if contact["is_spam"]:
                item_text += " - SPAM"
            
            item = QListWidgetItem(item_text)
            
            if contact["is_spam"]:
                item.setForeground(QColor("red"))
            
            self.contacts_list.addItem(item)
    
    def show_add_contact_dialog(self):
        """Show dialog to add a new contact"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Contact")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Name input
        name_input = QLineEdit()
        name_input.setPlaceholderText("Contact Name")
        layout.addWidget(name_input)
        
        # Phone input
        phone_input = QLineEdit()
        phone_input.setPlaceholderText("Phone Number")
        layout.addWidget(phone_input)
        
        # Spam checkbox
        spam_check = QComboBox()
        spam_check.addItem("Normal Contact")
        spam_check.addItem("Mark as Spam")
        layout.addWidget(spam_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.add_contact(
            name_input.text(),
            phone_input.text(),
            spam_check.currentIndex() == 1,
            dialog
        ))
        button_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def add_contact(self, name, phone, is_spam, dialog):
        """Add a new contact to the database"""
        if not name or not phone:
            QMessageBox.warning(self, "Error", "Name and phone number are required")
            return
        
        success = self.db.add_contact(phone, name, is_spam)
        if success:
            dialog.accept()
            self.search_contacts(self.contact_search.text())
            QMessageBox.information(self, "Success", f"Contact {name} added successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to add contact")
    
    def reset_models(self):
        """Reset the CallGuard AI models"""
        reply = QMessageBox.question(
            self, 
            "Reset Models", 
            "Are you sure you want to reset the detection models?\nThis will retrain them with default settings.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.callguard.deepfake_model._train_default_model()
            self.callguard.recording_model._train_default_model()
            QMessageBox.information(self, "Success", "Models have been reset and retrained")
    
    def simulate_normal_call(self):
        """Simulate a normal call (for demo purposes)"""
        self.phone_input.setText("+919876543210")
        self.start_call()
        
        # Simulate normal call statistics
        def update_sim():
            self.callguard.current_stats["deepfake_probabilities"] = [
                np.random.uniform(0.05, 0.3) for _ in range(10)
            ]
            self.callguard.current_stats["recording_probabilities"] = [
                np.random.uniform(0.05, 0.25) for _ in range(10)
            ]
        
        # Start simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(update_sim)
        self.sim_timer.start(1000)
        
        # Auto end call after 10 seconds
        QTimer.singleShot(10000, self.end_sim_call)
    
    def simulate_deepfake_call(self):
        """Simulate a deepfake call (for demo purposes)"""
        self.phone_input.setText("+919876543210")  # Use a spam number
        self.start_call()
        
        # Simulate deepfake call statistics
        def update_sim():
            self.callguard.current_stats["deepfake_probabilities"] = [
                np.random.uniform(0.75, 0.95) for _ in range(10)
            ]
            self.callguard.current_stats["recording_probabilities"] = [
                np.random.uniform(0.2, 0.4) for _ in range(10)
            ]
        
        # Start simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(update_sim)
        self.sim_timer.start(1000)
        
        # Auto end call after 10 seconds
        QTimer.singleShot(10000, self.end_sim_call)
    
    def simulate_recording_call(self):
        """Simulate a recorded call (for demo purposes)"""
        self.phone_input.setText("+919876543210")  # Use a spam number
        self.start_call()
        
        # Simulate recording call statistics
        def update_sim():
            self.callguard.current_stats["deepfake_probabilities"] = [
                np.random.uniform(0.3, 0.5) for _ in range(10)
            ]
            self.callguard.current_stats["recording_probabilities"] = [
                np.random.uniform(0.75, 0.95) for _ in range(10)
            ]
        
        # Start simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(update_sim)
        self.sim_timer.start(1000)
        
        # Auto end call after 10 seconds
        QTimer.singleShot(10000, self.end_sim_call)
    
    def end_sim_call(self):
        """End a simulated call"""
        if hasattr(self, 'sim_timer') and self.sim_timer.isActive():
            self.sim_timer.stop()
        self.end_call()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = TrueBlockMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()