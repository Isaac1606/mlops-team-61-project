#!/usr/bin/env python
"""
Script mejorado para verificar rangos de features con detección de outliers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.config import ConfigLoader, ProjectPaths
from src.data import DataLoader

def safe_convert_to_numeric(series):
    """Convierte una serie a numérico de forma segura."""
    # Primero intenta convertir directamente
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception:
        # Si falla, intenta limpiar strings vacíos y espacios
        cleaned = series.replace(['', ' ', '\n', '\t'], np.nan)
        return pd.to_numeric(cleaned, errors='coerce')

def detect_outliers(values, feature_name, z_threshold=3):
    """Detecta outliers usando Z-score."""
    if len(values) < 3:
        return pd.Series([False] * len(values), index=values.index)
    
    z_scores = np.abs((values - values.mean()) / values.std())
    outliers = z_scores > z_threshold
    return outliers

def analyze_feature_ranges(df: pd.DataFrame, dataset_name: str, features: list):
    """Analiza los rangos y estadísticas de features específicas con detección de outliers."""
    print(f"\n{'='*70}")
    print(f"ANÁLISIS: {dataset_name}")
    print(f"{'='*70}")
    print(f"Total filas: {len(df)}")
    print(f"Total columnas: {len(df.columns)}")
    
    print(f"\n{'Feature':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'Rango':<20} {'Estado':<20} {'Outliers':<15}")
    print("-" * 125)
    
    results = {}
    for feat in features:
        if feat in df.columns:
            # Convertir a numérico de forma segura
            values_series = safe_convert_to_numeric(df[feat])
            values = values_series.dropna()
            
            if len(values) > 0:
                # Detectar outliers
                outliers = detect_outliers(values, feat, z_threshold=3)
                n_outliers = outliers.sum()
                
                # Calcular estadísticas SIN outliers
                values_clean = values[~outliers] if n_outliers > 0 else values
                
                min_val = float(values.min())
                max_val = float(values.max())
                mean_val = float(values.mean())
                std_val = float(values.std())
                
                # Estadísticas sin outliers
                if len(values_clean) > 0:
                    min_clean = float(values_clean.min())
                    max_clean = float(values_clean.max())
                    mean_clean = float(values_clean.mean())
                else:
                    min_clean = min_val
                    max_clean = max_val
                    mean_clean = mean_val
                
                range_str = f"[{min_val:.4f}, {max_val:.4f}]"
                
                # Determinar si está normalizado (0-1) usando valores limpios
                is_normalized = (min_clean >= -0.1 and max_clean <= 1.1)
                
                status = '✅ Normalizado' if is_normalized else '❌ NO normalizado'
                
                outlier_info = f"{n_outliers} ({n_outliers/len(values)*100:.1f}%)" if n_outliers > 0 else "0"
                
                print(f"{feat:<20} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} {std_val:<12.4f} {range_str:<20} {status:<20} {outlier_info:<15}")
                
                # Mostrar detalles de outliers si existen
                if n_outliers > 0:
                    outlier_values = values[outliers].sort_values(ascending=False)
                    print(f"   ⚠️  Outliers detectados (top 5): {list(outlier_values.head(5).values)}")
                    print(f"   📊 Rango SIN outliers: [{min_clean:.4f}, {max_clean:.4f}]")
                
                results[feat] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'min_clean': min_clean,
                    'max_clean': max_clean,
                    'mean_clean': mean_clean,
                    'is_normalized': is_normalized,
                    'count': len(values),
                    'nulls': values_series.isnull().sum(),
                    'outliers': n_outliers,
                    'dtype': str(df[feat].dtype)
                }
            else:
                print(f"{feat:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'Sin datos':<20} {'⚠️ Todos NaN':<20} {'N/A':<15}")
        else:
            print(f"{feat:<20} {'NO EXISTE':<12}")
            # Mostrar columnas disponibles similares
            similar = [col for col in df.columns if feat.lower() in col.lower() or col.lower() in feat.lower()]
            if similar:
                print(f"   (Columnas similares encontradas: {', '.join(similar[:3])})")
    
    return results

def load_raw_data_safe(data_loader):
    """Carga datos raw de forma segura con manejo de errores."""
    try:
        # Intentar cargar directamente
        df = data_loader.load_raw_data()
        return df
    except Exception as e:
        print(f"⚠️ Error al cargar con DataLoader: {e}")
        print("   Intentando cargar directamente desde archivo...")
        
        try:
            # Cargar directamente desde el archivo
            config = ConfigLoader()
            paths = ProjectPaths(config)
            raw_file = paths.raw_data_file
            
            if not raw_file.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {raw_file}")
            
            # Cargar con opciones más permisivas
            df = pd.read_csv(
                raw_file,
                low_memory=False,
                na_values=['', ' ', 'NA', 'N/A', 'null', 'NULL', 'None'],
                keep_default_na=True
            )
            
            print(f"✅ Datos cargados directamente: {len(df)} filas, {len(df.columns)} columnas")
            
            # Mostrar tipos de datos
            print("\n📊 Tipos de datos en columnas relevantes:")
            for feat in ['temp', 'atemp', 'hum', 'windspeed']:
                if feat in df.columns:
                    sample_val = df[feat].iloc[0] if len(df) > 0 else 'N/A'
                    print(f"   {feat}: {df[feat].dtype} (ejemplo: {sample_val})")
            
            return df
        except Exception as e2:
            print(f"❌ Error también al cargar directamente: {e2}")
            raise

def main():
    """Función principal."""
    print("="*70)
    print("VERIFICACIÓN DE RANGOS DE FEATURES (CON DETECCIÓN DE OUTLIERS)")
    print("="*70)
    print("\nEste script verifica los rangos originales de las features")
    print("y detecta valores anómalos (outliers) que pueden distorsionar los rangos.\n")
    
    # Configuración
    config = ConfigLoader()
    paths = ProjectPaths(config)
    data_loader = DataLoader(paths)
    
    # Features a verificar
    features_to_check = ['temp', 'atemp', 'hum', 'windspeed']
    
    # ====================================================================
    # 1. ANÁLISIS DE DATOS RAW (originales, sin procesar)
    # ====================================================================
    raw_results = {}
    try:
        print("\n" + "="*70)
        print("1. CARGANDO DATOS RAW (originales)")
        print("="*70)
        df_raw = load_raw_data_safe(data_loader)
        
        raw_results = analyze_feature_ranges(df_raw, "DATOS RAW", features_to_check)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Saltando análisis de datos raw...")
    except Exception as e:
        print(f"❌ Error inesperado cargando datos raw: {e}")
        import traceback
        print("\n   Detalles del error:")
        traceback.print_exc()
        print("\n   Continuando con datos processed...")
    
    # ====================================================================
    # 2. ANÁLISIS DE DATOS PROCESSED (después de feature engineering)
    # ====================================================================
    processed_results = {}
    try:
        print("\n" + "="*70)
        print("2. CARGANDO DATOS PROCESSED (después de feature engineering)")
        print("="*70)
        df_train = data_loader.load_processed_data("train", normalized=False)
        
        processed_results = analyze_feature_ranges(df_train, "DATOS PROCESSED (TRAIN)", features_to_check)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Saltando análisis de datos processed...")
        print("   💡 Ejecuta primero: python src/data/make_dataset.py")
    except Exception as e:
        print(f"❌ Error inesperado cargando datos processed: {e}")
        import traceback
        print("\n   Detalles del error:")
        traceback.print_exc()
    
    # ====================================================================
    # 3. RESUMEN Y RECOMENDACIONES
    # ====================================================================
    print("\n" + "="*70)
    print("3. RESUMEN Y RECOMENDACIONES")
    print("="*70)
    
    if raw_results and processed_results:
        print("\n📊 Comparación RAW vs PROCESSED (SIN outliers):")
        print(f"{'Feature':<20} {'RAW Rango (limpio)':<25} {'PROCESSED Rango (limpio)':<25} {'Cambio':<20}")
        print("-" * 90)
        
        for feat in features_to_check:
            if feat in raw_results and feat in processed_results:
                raw_range = f"[{raw_results[feat]['min_clean']:.3f}, {raw_results[feat]['max_clean']:.3f}]"
                proc_range = f"[{processed_results[feat]['min_clean']:.3f}, {processed_results[feat]['max_clean']:.3f}]"
                
                # Verificar si cambió
                if (abs(raw_results[feat]['min_clean'] - processed_results[feat]['min_clean']) < 0.001 and
                    abs(raw_results[feat]['max_clean'] - processed_results[feat]['max_clean']) < 0.001):
                    cambio = "✅ Sin cambio"
                else:
                    cambio = "⚠️ Cambió"
                
                print(f"{feat:<20} {raw_range:<25} {proc_range:<25} {cambio:<20}")
    
    # Determinar si los datos están normalizados
    print("\n📋 CONCLUSIÓN:")
    print("-" * 70)
    
    # Priorizar processed_results si está disponible, sino usar raw_results
    results_to_use = processed_results if processed_results else raw_results
    
    if results_to_use:
        all_normalized = all(
            results_to_use[feat]['is_normalized'] 
            for feat in features_to_check 
            if feat in results_to_use
        )
        
        dataset_type = "PROCESSED" if processed_results else "RAW"
        
        if all_normalized:
            print(f"✅ Los datos {dataset_type} están NORMALIZADOS (rango 0-1) después de filtrar outliers")
            print("\n   Esto significa que:")
            print("   - Los datos originales ya vienen normalizados, O")
            print("   - Se normalizaron durante el feature engineering")
            print("\n   📝 RECOMENDACIÓN:")
            print("   - La API actual (que espera valores 0-1) es CORRECTA")
            print("   - Documenta claramente que los valores deben estar en rango [0, 1]")
            print("   - El preprocessor aplicará RobustScaler adicional (consistente con entrenamiento)")
        else:
            print(f"❌ Los datos {dataset_type} NO están normalizados (rango fuera de 0-1)")
            print("\n   Esto significa que:")
            print("   - Los datos están en escala original (ej: temp en Celsius)")
            print("   - El preprocessor los normalizará automáticamente")
            print("\n   📝 RECOMENDACIÓN:")
            print("   - La API debería aceptar valores ORIGINALES (no normalizados)")
            print("   - El usuario NO debería tener que normalizar manualmente")
            print("   - Actualiza API_EXAMPLES.md con rangos originales")
            
            # Mostrar rangos esperados (sin outliers)
            print("\n   📊 Rangos esperados en la API (sin outliers):")
            for feat in features_to_check:
                if feat in results_to_use:
                    r = results_to_use[feat]
                    print(f"   - {feat}: [{r['min_clean']:.2f}, {r['max_clean']:.2f}]")
    else:
        print("⚠️ No se pudieron cargar los datos para análisis")
        print("   Verifica que los archivos existan y sean accesibles")
    
    # Información sobre outliers
    if raw_results:
        print("\n" + "="*70)
        print("4. INFORMACIÓN SOBRE OUTLIERS")
        print("="*70)
        total_outliers = sum(raw_results[feat].get('outliers', 0) for feat in features_to_check if feat in raw_results)
        if total_outliers > 0:
            print(f"\n⚠️ Se detectaron {total_outliers} valores anómalos (outliers) en los datos RAW.")
            print("   Estos valores son filtrados automáticamente por DataCleaner._validate_and_filter_ranges()")
            print("   durante el procesamiento (desde la versión actualizada).")
            print("   Los rangos reportados arriba (SIN outliers) son los que realmente se usan.")
        else:
            print("\n✅ No se detectaron outliers significativos en los datos RAW.")
    
    # Información adicional sobre el preprocessor
    print("\n" + "="*70)
    print("5. INFORMACIÓN SOBRE EL PREPROCESSOR")
    print("="*70)
    print("\nEl pipeline incluye un preprocessor que usa RobustScaler.")
    print("Esto significa que:")
    print("  - Si los datos ya están en 0-1: RobustScaler los transformará")
    print("    (pero será consistente con el entrenamiento)")
    print("  - Si los datos están en escala original: RobustScaler los normalizará")
    print("    automáticamente usando estadísticas del entrenamiento")
    print("\n✅ En ambos casos, el preprocessor maneja la normalización correctamente.")
    
    print("\n" + "="*70)
    print("VERIFICACIÓN COMPLETA")
    print("="*70)

if __name__ == "__main__":
    main()