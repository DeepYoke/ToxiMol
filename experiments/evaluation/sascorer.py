#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import math
import pickle
import os.path as op
import gzip

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

def readFragmentScores(name="fpscores.pkl.gz"):
    """
    Read fragment scores from a pickle file
    """
    global _fscores
    # generate the full path filename:
    if name == "fpscores.pkl.gz":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open(name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    """
    Calculate the number of bridgeheads and spiro atoms in a molecule
    """
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    """
    Calculate the synthetic accessibility score for a molecule
    
    Args:
        m: RDKit molecule
        
    Returns:
        float: synthetic accessibility score between 1 (easy to synthesize) and 10 (very difficult)
    """
    if not m.GetNumAtoms():
        return None
        
    if _fscores is None:
        readFragmentScores()
        
    # fragment score
    sfp = mfpgen.GetSparseCountFingerprint(m)
    score1 = 0.
    nf = 0
    nze = sfp.GetNonzeroElements()
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count
    score1 /= nf
    
    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1
            
    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    
    # ---------------------------------------
    # This differs from the paper, which defines:
    # macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)
        
    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    
    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    numBits = len(nze)
    if nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * .5
        
    sascore = score1 + score2 + score3
    
    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
        
    return sascore

def processMols(mols):
    """
    Process a list of molecules and print their synthetic accessibility scores
    """
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue
            
        s = calculateScore(m)
        smiles = Chem.MolToSmiles(m)
        if s is None:
            print(f"{smiles}\t{m.GetProp('_Name')}\t{s}")
        else:
            print(f"{smiles}\t{m.GetProp('_Name')}\t{s:3f}")

if __name__ == '__main__':
    import sys
    import time
    
    t1 = time.time()
    if len(sys.argv) == 2:
        readFragmentScores()
    else:
        readFragmentScores(sys.argv[2])
        
    t2 = time.time()
    
    molFile = sys.argv[1]
    if molFile.endswith("smi"):
        suppl = Chem.SmilesMolSupplier(molFile)
    elif molFile.endswith("sdf"):
        suppl = Chem.SDMolSupplier(molFile)
    else:
        print(f"Unrecognized file extension for {molFile}")
        sys.exit()
        
    t3 = time.time()
    
    processMols(suppl)
    
    t4 = time.time()
    
    print('Reading took %.2f seconds. Calculating took %.2f seconds' % ((t2 - t1), (t4 - t3)),
          file=sys.stderr) 