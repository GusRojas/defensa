#!/usr/bin/env python3
"""
Script para comparar el rendimiento de modelos preentrenados durante el entrenamiento.
Adaptado para archivos training_history_MODELO.json con metadatos adicionales.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
import re
import matplotlib.gridspec as gridspec

# Paleta de colores fija por modelo (coherencia con TESTV2 y demás gráficas)
# Las claves coinciden con los nombres extraídos de training_history_<nombre>.json
MODEL_COLORS = {
    'PilotNetRegressor':            '#1f77b4',  # azul
    'MobileNetV3_large':            '#ff7f0e',  # naranja
    'MobileNetV3_small':            '#d62728',  # rojo
    'DroneNavSA-ConvLSTM_completo': '#bcbd22',  # oliva
    'ConvLSTM':                     '#2ca02c',  # verde
    'DroneResNet18':                '#17becf',  # cian
    'MLP':                          '#9467bd',  # morado
    'ConvMLP':                      '#8c564b',  # marrón
    'DroneNav-ConvLSTM':            '#e377c2',  # rosa
    'DroneNavSA-ConvLSTM':          '#7f7f7f',  # gris
}
_FALLBACK_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_model_color(model_name: str, idx: int = 0) -> str:
    """Devuelve el color canónico de un modelo, con fallback por índice."""
    return MODEL_COLORS.get(model_name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])

# Configurar estilo para gráficas académicas
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class ModelComparator:
    """Clase para comparar el rendimiento de múltiples modelos."""
    
    def __init__(self, models_dir: str = ".", output_dir: str = "comparison_results"):
        """
        Args:
            models_dir: Directorio que contiene los archivos training_history_*.json
            output_dir: Directorio donde guardar las gráficas
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.models_data = {}
        self.models_metadata = {}
        
    def extract_model_name(self, filename: str) -> str:
        """Extrae el nombre del modelo del nombre del archivo."""
        # Patrón: training_history_NOMBRE.json
        match = re.match(r'training_history_(.+)\.json', filename)
        if match:
            return match.group(1)
        return filename.replace('.json', '')
    
    def load_model_history(self, filepath: Path) -> tuple:
        """Carga el historial de entrenamiento de un modelo."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extraer historial de entrenamiento
            history = {}
            metadata = {}
            
            # Campos de entrenamiento
            training_fields = ['train_loss', 'val_loss', 'train_mae', 'val_mae', 
                             'train_mse', 'val_mse', 'train_accuracy', 'val_accuracy']
            
            for field in training_fields:
                if field in data and isinstance(data[field], list):
                    history[field] = data[field]
            
            # Campos de metadatos
            metadata_fields = ['batch_size', 'learning_rate', 'weight_decay', 
                              'epochs_trained', 'best_val_loss', 'total_time_hours']
            
            for field in metadata_fields:
                if field in data:
                    metadata[field] = data[field]
            
            return history, metadata
            
        except Exception as e:
            print(f"⚠️  Error cargando {filepath}: {e}")
            return None, None
    
    def discover_models(self, pattern: str = "training_history_*.json"):
        """Descubre automáticamente todos los archivos de historial."""
        print(f"🔍 Buscando archivos {pattern} en {self.models_dir}...")
        
        history_files = list(self.models_dir.glob(pattern))
        
        if not history_files:
            print(f"❌ No se encontraron archivos con patrón {pattern}")
            return False
        
        for filepath in history_files:
            model_name = self.extract_model_name(filepath.name)
            history, metadata = self.load_model_history(filepath)
            
            if history and 'train_loss' in history:
                self.models_data[model_name] = history
                self.models_metadata[model_name] = metadata
                
                epochs = len(history['train_loss'])
                has_val = 'val_loss' in history
                print(f"✓ Cargado: {model_name} ({epochs} épocas{', con validación' if has_val else ''})")
        
        print(f"\n📊 Total de modelos encontrados: {len(self.models_data)}")
        return len(self.models_data) > 0
    
    def plot_loss_comparison(self):
        """Genera gráfica comparativa de pérdidas de validación (val_loss)."""
        # Verificar si hay datos de validación
        has_val = any('val_loss' in hist for hist in self.models_data.values())
    
        if not has_val:
            print("⚠️ No hay datos de val_loss para graficar")
            return
    
        # Crear figura única
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Graficar val_loss de cada modelo
        for idx, (model_name, history) in enumerate(self.models_data.items()):
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], label=model_name, linewidth=2, alpha=0.8,
                        color=get_model_color(model_name, idx))
    
    # Configurar la gráfica
        ax.set_ylim(0.15, 0.3)
        ax.set_xlabel('Época')
        ax.set_ylabel('Pérdida (Loss)')
        ax.set_title('Pérdida de Validación')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.6)
    
        plt.tight_layout()
        output_path = self.output_dir / "loss_comparison.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()
    
    def plot_metrics_comparison(self):
        """Genera gráficas comparativas de métricas adicionales."""
        # Identificar todas las métricas disponibles (excluyendo loss)
        all_metrics = set()
        for history in self.models_data.values():
            all_metrics.update([k for k in history.keys() if 'loss' not in k.lower()])
        
        if not all_metrics:
            print("ℹ️  No se encontraron métricas adicionales además de loss")
            return
        
        # Crear subplots para cada métrica
        n_metrics = len(all_metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, metric in enumerate(sorted(all_metrics)):
            ax = axes[idx]

            for midx, (model_name, history) in enumerate(self.models_data.items()):
                if metric in history:
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(epochs, history[metric], label=model_name, linewidth=2, alpha=0.8,
                            color=get_model_color(model_name, midx))
            
            ax.set_xlabel('Época')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()
    
    def plot_training_validation_gap(self):
        """Gráfica de la brecha entre entrenamiento y validación (overfitting)."""
        has_val = any('val_loss' in hist for hist in self.models_data.values())
        
        if not has_val:
            print("ℹ️  No hay datos de validación para análisis de overfitting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, (model_name, history) in enumerate(self.models_data.items()):
            if 'val_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                gap = np.array(history['val_loss']) - np.array(history['train_loss'])
                ax.plot(epochs, gap, label=model_name, linewidth=2, alpha=0.8,
                        color=get_model_color(model_name, idx))
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(-0.1, 0.25)
        ax.set_xlabel('Época')
        ax.set_ylabel('Brecha (Val Loss - Train Loss)')
        ax.set_title('Brecha de Generalización (Overfitting)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / "overfitting_analysis.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()
    
    def create_summary_table(self):
        """Crea una tabla resumen con las mejores métricas de cada modelo."""
        summary = {}
        
        for model_name, history in self.models_data.items():
            metadata = self.models_metadata.get(model_name, {})
            
            summary[model_name] = {
                'best_train_loss': min(history['train_loss']),
                'final_train_loss': history['train_loss'][-1],
                'epochs': len(history['train_loss']),
                'batch_size': metadata.get('batch_size', 'N/A'),
                'learning_rate': metadata.get('learning_rate', 'N/A'),
                'total_time_hours': metadata.get('total_time_hours', 'N/A')
            }
            
            if 'val_loss' in history:
                summary[model_name]['best_val_loss'] = min(history['val_loss'])
                summary[model_name]['final_val_loss'] = history['val_loss'][-1]
        
        # Crear figura para la tabla
        fig, ax = plt.subplots(figsize=(14, len(summary) * 0.6 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        # Preparar datos para la tabla
        has_val = any('best_val_loss' in s for s in summary.values())
        
        if has_val:
            headers = ['Modelo', 'Mejor\nTrain Loss', 'Train Loss\nFinal', 
                      'Mejor\nVal Loss', 'Val Loss\nFinal', 'Épocas', 
                      'Batch\nSize', 'Learning\nRate', 'Tiempo\n(horas)']
        else:
            headers = ['Modelo', 'Mejor\nTrain Loss', 'Train Loss\nFinal', 
                      'Épocas', 'Batch\nSize', 'Learning\nRate', 'Tiempo\n(horas)']
        
        table_data = []
        
        for model_name, metrics in summary.items():
            if has_val:
                row = [
                    model_name[:30],  # Truncar nombres largos
                    f"{metrics['best_train_loss']:.6f}",
                    f"{metrics['final_train_loss']:.6f}",
                    f"{metrics.get('best_val_loss', 'N/A'):.6f}" if isinstance(metrics.get('best_val_loss'), (int, float)) else 'N/A',
                    f"{metrics.get('final_val_loss', 'N/A'):.6f}" if isinstance(metrics.get('final_val_loss'), (int, float)) else 'N/A',
                    str(metrics['epochs']),
                    str(metrics['batch_size']),
                    f"{metrics['learning_rate']:.6f}" if isinstance(metrics['learning_rate'], (int, float)) else str(metrics['learning_rate']),
                    f"{metrics['total_time_hours']:.2f}" if isinstance(metrics['total_time_hours'], (int, float)) else str(metrics['total_time_hours'])
                ]
            else:
                row = [
                    model_name[:30],
                    f"{metrics['best_train_loss']:.6f}",
                    f"{metrics['final_train_loss']:.6f}",
                    str(metrics['epochs']),
                    str(metrics['batch_size']),
                    f"{metrics['learning_rate']:.6f}" if isinstance(metrics['learning_rate'], (int, float)) else str(metrics['learning_rate']),
                    f"{metrics['total_time_hours']:.2f}" if isinstance(metrics['total_time_hours'], (int, float)) else str(metrics['total_time_hours'])
                ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                        cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Estilo de la tabla
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Resumen Comparativo de Modelos', fontsize=14, weight='bold', pad=20)
        
        output_path = self.output_dir / "summary_table.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"✓ Guardada: {output_path}")
        plt.close()
        
        # También guardar como JSON
        json_path = self.output_dir / "summary_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Guardado: {json_path}")
    
    def plot_learning_curves(self):
        """Gráficas de curvas de aprendizaje con train y val juntos."""
        n_models = len(self.models_data)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, history) in enumerate(self.models_data.items()):
            ax = axes[idx]
            epochs = range(1, len(history['train_loss']) + 1)

            ax.plot(epochs, history['train_loss'], label='Training', linewidth=2,
                    color='#1f77b4')
            if 'val_loss' in history:
                ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2,
                        color='#ff7f0e')

            ax.set_xlabel('Época')
            ax.set_ylim(0.05, 0.4)
            ax.set_ylabel('Pérdida (Loss)')
            ax.set_title(f'{model_name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.6)

        # Ocultar ejes vacíos
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        #plt.suptitle('Curvas de Aprendizaje por Modelo', fontsize=14, weight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "learning_curves.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()
    
    def plot_convergence_analysis(self):
        """Análisis de convergencia: primeras 100 épocas vs últimas 100."""
        n_models = len(self.models_data)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, history) in enumerate(self.models_data.items()):
            ax = axes[idx]
            total_epochs = len(history['train_loss'])
            color = get_model_color(model_name, idx)

            # Primeras 100 épocas
            early_epochs = min(100, total_epochs)
            ax.plot(range(1, early_epochs + 1),
                   history['train_loss'][:early_epochs],
                   label='Primeras épocas', linewidth=2, alpha=0.9, color=color)

            # Últimas 100 épocas
            if total_epochs > 100:
                late_start = total_epochs - 100
                ax.plot(range(late_start + 1, total_epochs + 1),
                       history['train_loss'][late_start:],
                       label='Últimas épocas', linewidth=2, alpha=0.5, color=color,
                       linestyle='--')

            ax.set_xlabel('Época')
            ax.set_ylabel('Training Loss')
            ax.set_title(f'{model_name} - Convergencia', color=color, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes vacíos
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Análisis de Convergencia', fontsize=14, weight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "convergence_analysis.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()

    def plot_combined_learning_curves(self):
        """Gráficas de curvas de aprendizaje individuales + comparación en una sola imagen."""
    #import matplotlib.gridspec as gridspec
    
        n_models = len(self.models_data)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols  # 4 filas para 10 modelos
    
        # Crear figura y grid
        fig = plt.figure(figsize=(15, 5 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
        # Graficar curvas individuales de cada modelo
        for idx, (model_name, history) in enumerate(self.models_data.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            epochs = range(1, len(history['train_loss']) + 1)

            ax.plot(epochs, history['train_loss'], label='Training', linewidth=2,
                    color='#1f77b4')
            if 'val_loss' in history:
                ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2,
                        color='#ff7f0e')

            ax.set_xlabel('Época')
            ax.set_ylim(0.05, 0.4)
            ax.set_ylabel('Pérdida (Loss)')
            ax.set_title(f'{model_name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.6)

        # Gráfica de comparación ocupando 2 columnas en la última fila
        has_val = any('val_loss' in hist for hist in self.models_data.values())

        if has_val:
            # Posición de la gráfica de comparación (última fila, columnas 1 y 2)
            ax_comparison = fig.add_subplot(gs[n_rows - 1, 1:])

            # Graficar val_loss de cada modelo
            for idx, (model_name, history) in enumerate(self.models_data.items()):
                if 'val_loss' in history:
                    epochs = range(1, len(history['val_loss']) + 1)
                    ax_comparison.plot(epochs, history['val_loss'],
                                 label=model_name, linewidth=2, alpha=0.8,
                                 color=get_model_color(model_name, idx))
        
            # Configurar la gráfica de comparación
            ax_comparison.set_ylim(0.15, 0.3)
            ax_comparison.set_xlabel('Época')
            ax_comparison.set_ylabel('Pérdida (Loss)')
            ax_comparison.set_title('Comparación de Pérdida de Validación')
            ax_comparison.legend(loc='best', fontsize=8, ncol=2)
            ax_comparison.grid(True, alpha=0.6)
    
        plt.tight_layout()
    
        output_path = self.output_dir / "combined_learning_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardada: {output_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Genera todas las gráficas de comparación."""
        print("\n📈 Generando gráficas comparativas...")
        
        self.plot_loss_comparison()
        self.plot_metrics_comparison()
        self.plot_training_validation_gap()
        self.plot_learning_curves()
        self.plot_convergence_analysis()
        self.plot_combined_learning_curves()
        self.create_summary_table()
        
        print(f"\n✅ Todas las gráficas guardadas en: {self.output_dir}")
        print("\nGráficas generadas:")
        print("  • loss_comparison.png - Comparación de pérdidas")
        print("  • metrics_comparison.png - Comparación de métricas")
        print("  • overfitting_analysis.png - Análisis de overfitting")
        print("  • learning_curves.png - Curvas de aprendizaje individuales")
        print("  • convergence_analysis.png - Análisis de convergencia")
        print("  • summary_table.png - Tabla resumen con metadatos")
        print("  • summary_metrics.json - Métricas en JSON")


def main():
    parser = argparse.ArgumentParser(
        description='Comparar rendimiento de modelos preentrenados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python compare_models_v2.py --models-dir ./
  python compare_models_v2.py --models-dir ./histories --output-dir ./thesis_figures
  python compare_models_v2.py --pattern "training_history_DroneNav*.json"

Archivos esperados:
  - training_history_MODELO1.json
  - training_history_MODELO2.json
  - training_history_MODELO3.json
  ...
        """
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='.',
        help='Directorio que contiene los archivos training_history_*.json (default: directorio actual)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='comparison_results',
        help='Directorio donde guardar las gráficas'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='training_history_*.json',
        help='Patrón de búsqueda de archivos (default: training_history_*.json)'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio
    if not Path(args.models_dir).exists():
        print(f"❌ Error: No se encontró el directorio {args.models_dir}")
        return
    
    # Crear comparador y ejecutar
    comparator = ModelComparator(args.models_dir, args.output_dir)
    
    if not comparator.discover_models(pattern=args.pattern):
        print("❌ No se encontraron archivos válidos para comparar")
        return
    
    comparator.generate_all_plots()
    
    print("\n🎓 ¡Listo para tu tesis!")


if __name__ == "__main__":
    main()
