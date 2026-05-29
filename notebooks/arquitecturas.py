import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=1, imu_dim=6, output_dim=6, hidden_dim=32):
        super().__init__()
        
        # ConvLSTM para imágenes
        self.conv_lstm = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 8, 8)),
        )
        
        # LSTM para datos IMU
        self.imu_lstm = nn.LSTM(input_size=imu_dim, hidden_size=32, num_layers=1, batch_first=True)
        
        # Calcular el tamaño correcto después de conv_lstm
        # hidden_dim * 1 * 8 * 8 = 32 * 1 * 8 * 8 = 2048
        conv_output_size = hidden_dim * 1 * 8 * 8
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_output_size + 32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim)
        )
    
    def forward(self, x_img, x_imu):
        # Procesar imágenes
        x_img = x_img.permute(0, 2, 1, 3, 4)
        x_img_feat = self.conv_lstm(x_img)
        x_img_feat = x_img_feat.view(x_img_feat.size(0), -1)
        
        # Procesar IMU
        _, (h_imu, _) = self.imu_lstm(x_imu)
        x_imu_feat = h_imu[-1]
        
        # Concatenar y pasar por FC
        x = torch.cat([x_img_feat, x_imu_feat], dim=1)
        return self.fc(x)
    

class DroneConvMLP(nn.Module):
    """
    Arquitectura híbrida para procesamiento de datos de drones.
    
    Esta red combina dos ramas:
    - img_branch: Procesa secuencias de imágenes usando convoluciones 3D
    - imu_branch: Procesa datos de sensores IMU usando capas lineales
    
    Ambas ramas se concatenan y pasan por capas totalmente conectadas.
    
    Args:
        image_size: Tupla (height, width) del tamaño de las imágenes. Default: (128, 128)
        num_frames: Número de frames en la secuencia temporal. Default: 10
        imu_features: Número de features por timestep del IMU. Default: 6
        output_size: Dimensión del vector de salida. Default: 6
        img_channels: Número de canales de entrada de la imagen. Default: 1
    """
    
    def __init__(self, image_size=(128, 128), num_frames=10, imu_features=6, 
                 output_size=6, img_channels=1):
        super(DroneConvMLP, self).__init__()
        
        self.image_size = image_size
        self.num_frames = num_frames
        self.imu_features = imu_features
        self.output_size = output_size
        self.img_channels = img_channels
        
        # Rama de procesamiento de imágenes (Conv3D)
        # Entrada: (batch, img_channels, num_frames, height, width)
        self.img_branch = nn.Sequential(
            nn.Conv3d(img_channels, 8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(5, 8, 8), stride=(5, 8, 8), padding=0)
        )
        
        # Calcular el tamaño de salida de img_branch
        img_feature_size = self._get_img_feature_size()
        
        # Rama de procesamiento de datos IMU (MLP)
        # Entrada: (batch, num_frames * imu_features)
        imu_input_size = num_frames * imu_features
        self.imu_branch = nn.Sequential(
            nn.Linear(in_features=imu_input_size, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU()
        )
        
        # Capas totalmente conectadas finales
        # Concatenar características de ambas ramas
        combined_features = img_feature_size + 32  # img_features + imu_features
        self.fc1 = nn.Linear(in_features=combined_features, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size, bias=True)
    
    def _get_img_feature_size(self):
        """
        Calcula el tamaño de las features después de img_branch.
        """
        # Crear un tensor dummy para calcular el tamaño de salida
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.img_channels, self.num_frames, 
                                     self.image_size[0], self.image_size[1])
            dummy_output = self.img_branch(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def forward(self, img, imu):
        """
        Forward pass del modelo.
        
        Args:
            img: Tensor de imágenes con forma:
                - (batch, num_frames, img_channels, height, width) 
                - (num_frames, img_channels, height, width) para una sola muestra
            imu: Tensor de datos IMU con forma:
                - (batch, num_frames, imu_features)
                - (num_frames, imu_features) para una sola muestra
        
        Returns:
            output: Tensor de salida con forma (batch, output_size)
        """
        batch_mode = True
        
        # Detectar si es una sola muestra o un batch
        if img.dim() == 4:  # (num_frames, img_channels, height, width)
            img = img.unsqueeze(0)  # (1, num_frames, img_channels, height, width)
            batch_mode = False
        
        if imu.dim() == 2:  # (num_frames, imu_features)
            imu = imu.unsqueeze(0)  # (1, num_frames, imu_features)
        
        # Reordenar imágenes de (batch, num_frames, channels, H, W) a (batch, channels, num_frames, H, W)
        img = img.permute(0, 2, 1, 3, 4)
        
        # Aplanar IMU de (batch, num_frames, imu_features) a (batch, num_frames * imu_features)
        batch_size = imu.size(0)
        imu = imu.view(batch_size, -1)
        
        # Procesar rama de imágenes
        img_features = self.img_branch(img)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten
        
        # Procesar rama de IMU
        imu_features = self.imu_branch(imu)
        
        # Concatenar características de ambas ramas
        combined = torch.cat([img_features, imu_features], dim=1)
        
        # Capas finales
        x = torch.relu(self.fc1(combined))
        output = self.fc2(x)
        
        # Si era una sola muestra, remover dimensión batch
        if not batch_mode:
            output = output.squeeze(0)
        
        return output
    
class DroneMLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 10 * 1 * 128 * 128 + 10 * 6  # 163840 + 60 = 163900
        hidden_dim = 512
        output_dim = 5 * 6  # 5 pasos de tiempo × 6 características

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_img, x_imu):
        x_img_flat = x_img.view(x_img.size(0), -1)  # [256, 163900]
        x_imu_flat = x_imu.view(x_imu.size(0), -1)  # [256, 60]
        x = torch.cat([x_img_flat, x_imu_flat], dim=1)  # [256, 10300]
        x = F.relu(self.fc1(x))
        y = self.fc3(x)
        # Reorganizar salida a [batch, 5, 6]
        y = y.view(x.size(0), 5, 6)
        return y
    
class PilotNetRegressor(nn.Module):
    def __init__(self, image_size=(128, 128), sequence_length=10, imu_features=6, output_size=6):
        """
        Args:
            image_size: tuple (height, width) del tamaño de las imágenes de entrada
            sequence_length: longitud de la secuencia temporal (no afecta la arquitectura, solo para referencia)
            imu_features: número de características de IMU (default: 6)
            output_size: número de valores de salida (default: 6 para roll, pitch, yaw, acc_x, acc_y, acc_z)
        """
        super(PilotNetRegressor, self).__init__()
        
        self.image_size = image_size
        self.imu_features = imu_features
        self.output_size = output_size
        
        # Capas Convolucionales (similar a PilotNet)
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.batch_norm2 = nn.BatchNorm2d(36)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(64)
        
        # Calcular dinámicamente el tamaño de salida de las convoluciones
        self.cnn_output_size = self._get_conv_output_size(image_size)
        
        # Procesamiento temporal de secuencias
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Procesamiento de IMU
        self.imu_fc = nn.Sequential(
            nn.Linear(imu_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Capas densas para fusión y regresión
        self.fc1 = nn.Linear(512 + 64, 128)  # LSTM output + IMU features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
        self.dropout = nn.Dropout(0.3)
    
    def _get_conv_output_size(self, image_size):
        """
        Calcula el tamaño de salida de las capas convolucionales
        haciendo un forward pass con un tensor dummy
        """
        with torch.no_grad():
            # Crear un tensor dummy
            dummy_input = torch.zeros(1, 1, image_size[0], image_size[1])
            
            # Pasar por las capas convolucionales
            x = F.relu(self.batch_norm1(self.conv1(dummy_input)))
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = F.relu(self.batch_norm3(self.conv3(x)))
            x = F.relu(self.batch_norm4(self.conv4(x)))
            x = F.relu(self.batch_norm5(self.conv5(x)))
            
            # Retornar el tamaño flatten
            return x.numel()
    
    def forward(self, x_img, x_imu):
        """
        Args:
            x_img: (batch_size, sequence_length, channels, height, width)
            x_imu: (batch_size, sequence_length, imu_features)
        Returns:
            output: (batch_size, output_size)
        """
        batch_size, seq_len, C, H, W = x_img.shape
        
        # Procesar todas las imágenes de la secuencia en paralelo
        x = x_img.view(batch_size * seq_len, C, H, W)
        
        # Capas convolucionales
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.relu(self.batch_norm5(self.conv5(x)))
        
        # Flatten
        x = x.view(batch_size * seq_len, -1)
        
        # Reshape para secuencia temporal
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM para capturar dependencias temporales
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Tomar el último estado oculto
        x_temporal = lstm_out[:, -1, :]
        
        # Procesar IMU (usar el último frame de la secuencia)
        x_imu_last = x_imu[:, -1, :]  # (batch_size, imu_features)
        x_imu_features = self.imu_fc(x_imu_last)
        
        # Fusionar características de imagen e IMU
        x_fused = torch.cat([x_temporal, x_imu_features], dim=1)
        
        # Capas densas para regresión
        x = F.relu(self.fc1(x_fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        
        return output

class DroneResNet(nn.Module):
    def __init__(self, resnet_type='resnet18', pretrained=False):
        """
        Arquitectura ResNet + Regresión para control de drone
        
        Args:
            resnet_type: 'resnet18' o 'resnet34'
            pretrained: usar pesos preentrenados (se adaptan de 3 a 1 canal)
        """
        super(DroneResNet, self).__init__()
        
        # ============ PROCESAMIENTO DE IMÁGENES ============
        # Cargar ResNet base
        if resnet_type == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        else:  # resnet34
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        
        # Adaptar primera capa para 1 canal (monocromático)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Promediar pesos RGB para inicializar el canal monocromático
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Capas convolucionales de ResNet (sin la última FC)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # ============ PROCESAMIENTO TEMPORAL ============
        # LSTM para agregar información temporal de las imágenes
        self.lstm_visual = nn.LSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # ============ PROCESAMIENTO DE IMU ============
        # Red para procesar secuencias IMU
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM para agregar información temporal de IMU
        self.lstm_imu = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # ============ FUSIÓN Y REGRESIÓN ============
        # Combinar características visuales + IMU
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabezal de regresión final
        self.regressor = nn.Linear(64, 6)  # [roll, pitch, yaw, acc_x, acc_y, acc_z]
        
    def forward(self, x_img, x_imu):
        """
        Args:
            x_img: (batch_size, seq_len, 1, 128, 128)
            x_imu: (batch_size, seq_len, 6)
        Returns:
            output: (batch_size, 6)
        """
        batch_size, seq_len, C, H, W = x_img.shape
        
        # ============ EXTRACCIÓN DE CARACTERÍSTICAS VISUALES ============
        # Procesar cada frame con ResNet
        x_img = x_img.view(batch_size * seq_len, C, H, W)
        
        # Forward por ResNet
        x = self.conv1(x_img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        visual_features = torch.flatten(x, 1)  # (batch*seq, 512)
        
        # Reshape para LSTM: (batch, seq, features)
        visual_features = visual_features.view(batch_size, seq_len, -1)
        
        # Agregación temporal con LSTM
        lstm_out, (h_n, c_n) = self.lstm_visual(visual_features)
        visual_encoded = h_n[-1]  # Último estado oculto (batch, 256)
        
        # ============ PROCESAMIENTO DE IMU ============
        # Codificar cada timestep de IMU
        imu_encoded = self.imu_encoder(x_imu)  # (batch, seq, 128)
        
        # Agregación temporal con LSTM
        lstm_imu_out, (h_imu, c_imu) = self.lstm_imu(imu_encoded)
        imu_encoded = h_imu[-1]  # Último estado oculto (batch, 128)
        
        # ============ FUSIÓN Y REGRESIÓN ============
        # Concatenar características visuales e IMU
        fused = torch.cat([visual_encoded, imu_encoded], dim=1)  # (batch, 384)
        
        # Capas de fusión
        fused = self.fusion(fused)
        
        # Regresión final
        output = self.regressor(fused)  # (batch, 6)
        
        return output
    
class DroneMobileNetV3(nn.Module):
    def __init__(self, mobilenet_type='small', weights=None):
        """
        Arquitectura MobileNetV3 + LSTM para control de drone
        
        Args:
            mobilenet_type: 'small' o 'large'
            pretrained: usar pesos preentrenados (se adaptan de 3 a 1 canal)
        """
        super(DroneMobileNetV3, self).__init__()
        
        # ============ PROCESAMIENTO DE IMÁGENES ============
        # Cargar MobileNetV3 base
        if mobilenet_type == 'small':
            # MobileNetV3-Small: más ligero (~1.5M params)
            mobilenet = models.mobilenet_v3_small(pretrained=weights)
            feature_dim = 576  # Dimensión de salida antes del clasificador
        else:  # 'large'
            # MobileNetV3-Large: más preciso (~4.2M params)
            mobilenet = models.mobilenet_v3_large(pretrained=weights)
            feature_dim = 960  # Dimensión de salida antes del clasificador
        
        # Adaptar primera capa para 1 canal (imágenes monocromáticas)
        # MobileNetV3 usa Conv2d como primera capa en features[0][0]
        original_conv = mobilenet.features[0][0]
        self.first_conv = nn.Conv2d(
            in_channels=1,  # 1 canal en lugar de 3
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        if weights:
            # Promediar pesos RGB para inicializar el canal monocromático
            with torch.no_grad():
                self.first_conv.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Copiar el resto de las features de MobileNetV3
        self.features = nn.Sequential(
            self.first_conv,
            *list(mobilenet.features.children())[1:]  # Resto de capas
        )
        
        # Pooling adaptativo para obtener features de tamaño fijo
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ============ PROCESAMIENTO TEMPORAL ============
        # LSTM para agregar información temporal de las imágenes
        self.lstm_visual = nn.LSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # ============ PROCESAMIENTO DE IMU ============
        # Red para procesar secuencias IMU
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM para agregar información temporal de IMU
        self.lstm_imu = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # ============ FUSIÓN Y REGRESIÓN ============
        # Combinar características visuales + IMU (256 + 128 = 384)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabezal de regresión final
        self.regressor = nn.Linear(64, 6)  # [roll, pitch, yaw, acc_x, acc_y, acc_z]
        
    def forward(self, x_img, x_imu):
        """
        Args:
            x_img: (batch_size, seq_len, 1, 128, 128)
            x_imu: (batch_size, seq_len, 6)
        Returns:
            output: (batch_size, 6)
        """
        batch_size, seq_len, C, H, W = x_img.shape
        
        # ============ EXTRACCIÓN DE CARACTERÍSTICAS VISUALES ============
        # Procesar cada frame con MobileNetV3
        x_img = x_img.view(batch_size * seq_len, C, H, W)
        
        # Forward por MobileNetV3
        x = self.features(x_img)  # (batch*seq, feature_dim, H', W')
        x = self.avgpool(x)  # (batch*seq, feature_dim, 1, 1)
        visual_features = torch.flatten(x, 1)  # (batch*seq, feature_dim)
        
        # Reshape para LSTM: (batch, seq, features)
        visual_features = visual_features.view(batch_size, seq_len, -1)
        
        # Agregación temporal con LSTM
        lstm_out, (h_n, c_n) = self.lstm_visual(visual_features)
        visual_encoded = h_n[-1]  # Último estado oculto (batch, 256)
        
        # ============ PROCESAMIENTO DE IMU ============
        # Codificar cada timestep de IMU
        imu_encoded = self.imu_encoder(x_imu)  # (batch, seq, 128)
        
        # Agregación temporal con LSTM
        lstm_imu_out, (h_imu, c_imu) = self.lstm_imu(imu_encoded)
        imu_encoded = h_imu[-1]  # Último estado oculto (batch, 128)
        
        # ============ FUSIÓN Y REGRESIÓN ============
        # Concatenar características visuales e IMU
        fused = torch.cat([visual_encoded, imu_encoded], dim=1)  # (batch, 384)
        
        # Capas de fusión
        fused = self.fusion(fused)
        
        # Regresión final
        output = self.regressor(fused)  # (batch, 6)
        
        return output

class SelfAttentionMemory(nn.Module):
    """
    Self-Attention Memory module from SA-ConvLSTM paper
    Captures long-range spatial and temporal dependencies
    """
    def __init__(self, input_channels, hidden_channels, reduction=8):
        super(SelfAttentionMemory, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.reduction = reduction
        
        # Query, Key, Value for hidden state h_t
        self.Wq_h = nn.Conv2d(hidden_channels, hidden_channels // reduction, 
                              kernel_size=1, bias=False)
        self.Wk_h = nn.Conv2d(hidden_channels, hidden_channels // reduction, 
                              kernel_size=1, bias=False)
        self.Wv_h = nn.Conv2d(hidden_channels, hidden_channels, 
                              kernel_size=1, bias=False)
        
        # Key, Value for memory M_{t-1}
        self.Wk_m = nn.Conv2d(hidden_channels, hidden_channels // reduction, 
                              kernel_size=1, bias=False)
        self.Wv_m = nn.Conv2d(hidden_channels, hidden_channels, 
                              kernel_size=1, bias=False)
        
        # Fusion of Z_h and Z_m
        self.Wz = nn.Conv2d(hidden_channels * 2, hidden_channels, 
                           kernel_size=1, bias=True)
        
        # Memory update gates (using depthwise separable convolutions)
        # Input gate
        self.Wmi = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.Whi = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.bi = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        # Candidate gate
        self.Wmg = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.Whg = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.bg = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        # Output gate
        self.Wmo = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.Who = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        )
        self.bo = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        # Residual connection weight
        self.Wf = nn.Conv2d(hidden_channels, hidden_channels, 
                           kernel_size=1, bias=True)
        
    def forward(self, h_t, m_prev):
        """
        Args:
            h_t: Current hidden state [B, C, H, W]
            m_prev: Previous memory state [B, C, H, W]
        
        Returns:
            h_out: Output hidden state [B, C, H, W]
            m_t: Updated memory state [B, C, H, W]
        """
        batch_size, channels, height, width = h_t.shape
        N = height * width
        
        # ========== Feature Aggregation ==========
        
        # Self-attention on h_t
        Q_h = self.Wq_h(h_t).view(batch_size, -1, N)  # [B, C', N]
        K_h = self.Wk_h(h_t).view(batch_size, -1, N)  # [B, C', N]
        V_h = self.Wv_h(h_t).view(batch_size, channels, N)  # [B, C, N]
        
        # Attention weights for h_t (Eq. 2-3 in paper)
        attention_h = torch.bmm(Q_h.transpose(1, 2), K_h)  # [B, N, N]
        attention_h = F.softmax(attention_h, dim=-1)
        
        # Aggregated features from h_t (Eq. 4)
        Z_h = torch.bmm(V_h, attention_h.transpose(1, 2))  # [B, C, N]
        Z_h = Z_h.view(batch_size, channels, height, width)
        
        # Cross-attention with memory
        K_m = self.Wk_m(m_prev).view(batch_size, -1, N)  # [B, C', N]
        V_m = self.Wv_m(m_prev).view(batch_size, channels, N)  # [B, C, N]
        
        # Attention weights between h_t and m_prev (Eq. 5-6)
        attention_m = torch.bmm(Q_h.transpose(1, 2), K_m)  # [B, N, N]
        attention_m = F.softmax(attention_m, dim=-1)
        
        # Aggregated features from memory (Eq. 7)
        Z_m = torch.bmm(V_m, attention_m.transpose(1, 2))  # [B, C, N]
        Z_m = Z_m.view(batch_size, channels, height, width)
        
        # Fuse Z_h and Z_m
        Z = self.Wz(torch.cat([Z_h, Z_m], dim=1))  # [B, C, H, W]
        
        # ========== Memory Updating (Eq. 8) ==========
        
        # Input gate
        i_t = torch.sigmoid(self.Wmi(Z) + self.Whi(h_t) + self.bi)
        
        # Candidate gate
        g_t = torch.tanh(self.Wmg(Z) + self.Whg(h_t) + self.bg)
        
        # Update memory (using 1 - i_t as forget gate)
        m_t = (1 - i_t) * m_prev + i_t * g_t
        
        # ========== Output (Eq. 9) ==========
        
        # Output gate
        o_t = torch.sigmoid(self.Wmo(Z) + self.Who(h_t) + self.bo)
        
        # Final output with residual connection
        h_out = o_t * m_t
        h_out = self.Wf(h_out) + h_t  # Residual connection
        
        return h_out, m_t


class LightweightSAM(nn.Module):
    """
    Lightweight version of SAM for embedded deployment
    Reduces spatial resolution before attention computation
    """
    def __init__(self, input_channels, hidden_channels, 
                 reduction=8, spatial_reduction=2):
        super(LightweightSAM, self).__init__()
        
        self.spatial_reduction = spatial_reduction
        
        # Downsampling before attention
        if spatial_reduction > 1:
            self.downsample = nn.AvgPool2d(kernel_size=spatial_reduction, 
                                          stride=spatial_reduction)
            self.upsample = nn.Upsample(scale_factor=spatial_reduction, 
                                       mode='bilinear', align_corners=False)
        else:
            self.downsample = nn.Identity()
            self.upsample = nn.Identity()
        
        # Core SAM module
        self.sam = SelfAttentionMemory(input_channels, hidden_channels, reduction)
        
    def forward(self, h_t, m_prev):
        # Downsample for efficient attention
        h_small = self.downsample(h_t)
        m_small = self.downsample(m_prev)
        
        # Apply attention at reduced resolution
        h_out_small, m_out_small = self.sam(h_small, m_small)
        
        # Upsample back to original resolution
        h_out = self.upsample(h_out_small)
        m_out = self.upsample(m_out_small)
        
        return h_out, m_out
    
class SAConvLSTMCell(nn.Module):
    """
    Self-Attention ConvLSTM Cell
    Combines standard ConvLSTM with Self-Attention Memory module
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3,
                 use_sam=True, lightweight=False, spatial_reduction=2):
        super(SAConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_sam = use_sam
        
        # Standard ConvLSTM gates
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=False)
        
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=False)
        
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=False)
        
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                            padding=self.padding, bias=False)
        
        # Self-Attention Memory module
        if use_sam:
            if lightweight:
                self.sam = LightweightSAM(input_channels, hidden_channels,
                                         spatial_reduction=spatial_reduction)
            else:
                self.sam = SelfAttentionMemory(input_channels, hidden_channels)
        
    def forward(self, x_t, hidden_state):
        """
        Args:
            x_t: Input tensor [B, C_in, H, W]
            hidden_state: Tuple of (h, c, m) or (h, c) or None
        
        Returns:
            h_next: Next hidden state [B, C_hidden, H, W]
            hidden_state: Tuple of updated states
        """
        if hidden_state is None:
            batch_size, _, height, width = x_t.shape
            h_prev = torch.zeros(batch_size, self.hidden_channels, 
                                height, width, device=x_t.device)
            c_prev = torch.zeros(batch_size, self.hidden_channels, 
                                height, width, device=x_t.device)
            if self.use_sam:
                m_prev = torch.zeros(batch_size, self.hidden_channels, 
                                    height, width, device=x_t.device)
        else:
            if self.use_sam:
                h_prev, c_prev, m_prev = hidden_state
            else:
                h_prev, c_prev = hidden_state
        
        # Apply Self-Attention Memory if enabled
        if self.use_sam:
            h_att, m_next = self.sam(h_prev, m_prev)
        else:
            h_att = h_prev
            m_next = None
        
        # Standard ConvLSTM operations with attention-enhanced hidden state
        i_t = torch.sigmoid(self.Wxi(x_t) + self.Whi(h_att))
        f_t = torch.sigmoid(self.Wxf(x_t) + self.Whf(h_att))
        g_t = torch.tanh(self.Wxc(x_t) + self.Whc(h_att))
        o_t = torch.sigmoid(self.Wxo(x_t) + self.Who(h_att))
        
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)
        
        if self.use_sam:
            return h_next, (h_next, c_next, m_next)
        else:
            return h_next, (h_next, c_next)
        
class DroneNavSAConvLSTM(nn.Module):
    """
    SA-ConvLSTM network for drone autonomous navigation
    
    Input:
        - image_seq: [B, T, 1, 128, 128]  # Sequence of grayscale images
        - imu_seq: [B, T, 6]              # Sequence of IMU readings (ax, ay, az, gx, gy, gz)
    
    Output:
        - [B, 6]  # Predicted IMU for next timestep (3 angular velocities + 3 linear accelerations)
    """
    def __init__(self, 
                 image_channels=1,
                 imu_channels=6,
                 num_layers=4,
                 hidden_channels=64,
                 kernel_size=3,
                 use_sam=True,
                 lightweight=True,
                 output_dim=6):
        super(DroneNavSAConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.use_sam = use_sam
        self.output_dim = output_dim
        
        # Visual feature encoder (CNN backbone)
        self.visual_encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16 (reduced spatial size for efficiency)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # IMU encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        
        # Project IMU features to spatial format (16x16)
        self.imu_spatial_proj = nn.Sequential(
            nn.Linear(128, 16 * 16),
            nn.ReLU(inplace=True),
        )
        
        # Fusion: visual (64 channels) + IMU (1 channel) = 65 channels
        # Project to hidden_channels
        self.fusion_conv = nn.Conv2d(65, hidden_channels, kernel_size=1)
        
        # Stack of SA-ConvLSTM layers
        self.sa_convlstm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.sa_convlstm_layers.append(
                SAConvLSTMCell(
                    input_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    use_sam=use_sam,
                    lightweight=lightweight,
                    spatial_reduction=2 if lightweight else 1
                )
            )
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.GroupNorm(8, hidden_channels) for _ in range(num_layers)
        ])
        
        # IMU prediction head (predicts next IMU values)
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling [B, C, 16, 16] -> [B, C, 1, 1]
            nn.Flatten(),             # [B, C]
            nn.Linear(hidden_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)  # [B, 6] - no activation (raw IMU values)
        )
        
    def forward(self, image_seq, imu_seq):
        """
        Args:
            image_seq: [B, T, C, H, W] = [B, T, 1, 128, 128]
            imu_seq: [B, T, imu_dim] = [B, T, 6]
        
        Returns:
            output: [B, 6] - Predicted IMU for next timestep
        """
        batch_size, seq_len, C, H, W = image_seq.shape
        
        # Initialize hidden states for all layers
        hidden_states = [None] * self.num_layers
        
        # Process sequence temporally
        for t in range(seq_len):
            # Extract visual features
            visual_feat = self.visual_encoder(image_seq[:, t])  # [B, 64, 16, 16]
            
            # Extract IMU features
            imu_feat = self.imu_encoder(imu_seq[:, t])  # [B, 128]
            imu_spatial = self.imu_spatial_proj(imu_feat)  # [B, 256]
            imu_spatial = imu_spatial.view(batch_size, 1, 16, 16)  # [B, 1, 16, 16]
            
            # Fuse visual and IMU features
            x = torch.cat([visual_feat, imu_spatial], dim=1)  # [B, 65, 16, 16]
            x = self.fusion_conv(x)  # [B, hidden_channels, 16, 16]
            
            # Pass through SA-ConvLSTM layers
            for layer_idx in range(self.num_layers):
                x, hidden_states[layer_idx] = self.sa_convlstm_layers[layer_idx](
                    x, hidden_states[layer_idx]
                )
                x = self.layer_norms[layer_idx](x)
        
        # Decode final hidden state to IMU prediction
        # Only use the last timestep's hidden state
        output = self.decoder(x)  # [B, 6]
        
        return output