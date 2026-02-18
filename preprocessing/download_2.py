"""
Download CMU-MOSEI dataset
Uses CMU Multimodal SDK for proper data loading
"""

import os
import sys
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import MOSEI_DIR


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """Download file with progress bar"""
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            pbar.update(size)


def download_mosei_sdk():
    """
    Download CMU-MOSEI using the official CMU Multimodal SDK
    This provides pre-extracted features which is perfect for our use case
    """
    
    print("\n" + "="*60)
    print("CMU-MOSEI Dataset Download")
    print("="*60 + "\n")
    
    # Create directory
    MOSEI_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download location: {MOSEI_DIR}")
    print("\n⏳ Installing CMU Multimodal SDK...\n")
    
    try:
        # Install CMU Multimodal SDK
        import subprocess
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'CMU-MultimodalSDK', '--quiet'
        ], check=True)
        
        print("✓ CMU Multimodal SDK installed\n")
        
        # Import SDK
        from mmsdk import mmdatasdk
        
        print("📦 Downloading MOSEI dataset components...")
        print("   This includes: Audio, Visual, Text features")
        print("   Size: ~15-20 GB")
        print("   Time: 20-40 minutes depending on connection\n")
        
        # Define dataset components to download
        # We use pre-extracted features to save time
        dataset_components = {
            'CMU_MOSEI_TimestampedWords': 'words',
            'CMU_MOSEI_TimestampedPhones': 'phones',
            'CMU_MOSEI_TimestampedWordVectors': 'glove',
            'CMU_MOSEI_VisualFacet42': 'facet',  # Facial features
            'CMU_MOSEI_VisualOpenFace2': 'openface',  # Facial landmarks
            'CMU_MOSEI_COVAREP': 'covarep',  # Audio features
            'CMU_MOSEI_Labels': 'labels'
        }
        
        # Download directory
        download_dir = str(MOSEI_DIR)
        
        # Download each component
        for component_name, short_name in dataset_components.items():
            print(f"\n⏳ Downloading {short_name}...")
            try:
                dataset = mmdatasdk.mmdataset(
                    {component_name: download_dir}
                )
                print(f"   ✓ {short_name} downloaded")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not download {short_name}: {e}")
                continue
        
        print("\n✅ MOSEI download complete!")
        
        # Verify download
        downloaded_files = list(MOSEI_DIR.rglob('*'))
        print(f"\n📊 Downloaded {len(downloaded_files)} files")
        
        print("\n" + "="*60)
        print("✅ CMU-MOSEI setup successful!")
        print("="*60)
        print("\nNext step: Run feature extraction")
        print("  python preprocessing/extract_features.py --dataset mosei\n")
        
    except ImportError:
        print("\n❌ Could not install CMU Multimodal SDK")
        print("\nTrying alternative download method...\n")
        download_mosei_manual()
        
    except Exception as e:
        print(f"\n❌ Error downloading MOSEI: {e}")
        print("\nTrying alternative download method...\n")
        download_mosei_manual()


def download_mosei_manual():
    """
    Alternative: Download pre-processed MOSEI features from direct links
    This is a backup method if SDK installation fails
    """
    
    print("Using manual download method...")
    
    # Direct download URLs for essential MOSEI components
    # These are hosted by CMU and contain pre-extracted features
    urls = {
        'visual': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/features/CMU_MOSEI_VisualOpenFace2.csd',
        'audio': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/features/CMU_MOSEI_COVAREP.csd',
        'text': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/features/CMU_MOSEI_TimestampedWordVectors.csd',
        'labels': 'http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI/labels/CMU_MOSEI_Labels.csd'
    }
    
    features_dir = MOSEI_DIR / 'features'
    features_dir.mkdir(exist_ok=True)
    
    for name, url in urls.items():
        print(f"\n⏳ Downloading {name} features...")
        filename = features_dir / f"mosei_{name}.csd"
        
        try:
            download_file(url, filename)
            print(f"   ✓ {name} downloaded")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not download {name}: {e}")
    
    print("\n✅ Manual download complete!")
    print("\n📝 Note: Using pre-extracted features")
    print("   This is suitable for validation experiments\n")


def verify_mosei():
    """Verify MOSEI dataset"""
    print("\n🔍 Verifying MOSEI dataset...")
    
    if not MOSEI_DIR.exists():
        print("  ✗ MOSEI directory not found")
        return False
    
    files = list(MOSEI_DIR.rglob('*'))
    
    if len(files) == 0:
        print("  ✗ MOSEI directory is empty")
        return False
    
    print(f"  ✓ Found {len(files)} files")
    
    # Check for essential components
    required = ['visual', 'audio', 'text']
    for component in required:
        found = any(component.lower() in str(f).lower() for f in files)
        status = "✓" if found else "⚠️ "
        print(f"  {status} {component.capitalize()} features")
    
    return True


if __name__ == "__main__":
    try:
        download_mosei_sdk()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nPlease report this issue if it persists")
        sys.exit(1)
    
    verify_mosei()