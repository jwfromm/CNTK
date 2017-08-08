#include "../CPP/BinMul2A1BOp.h"
#include "../CPP/UserMatrixMultiplicationOp.h"

using namespace CNTK;

extern "C" 
#ifdef _WIN32
__declspec (dllexport)
#endif
Function* CreateBinGemm2A1B(const Variable* operands, size_t /*numOperands*/, const Dictionary* attributes, const wchar_t* name)
{
    return new BinMul2A1B(operands[0], operands[1], *attributes, name);
}
Function* CreateUserTimesFunction(const Variable* operands, size_t /*numOperands*/, const Dictionary* attributes, const wchar_t* name)
{
    return new UserTimesFunction(operands[0], operands[1], *attributes, name);
}
