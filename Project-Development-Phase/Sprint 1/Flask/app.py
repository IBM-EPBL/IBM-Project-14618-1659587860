from flask import Flask, render_template, request

import os
import cv2
import operator
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

