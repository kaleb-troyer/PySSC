
#include <unordered_map>

// double CEPCI_lookup(int year, int type) {


//     double factor = 0.0; // CEPCI for given year and type
//     switch (year) {
//         case 2012: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 584.6;
//                     break; 
//                 case HTEX: 
//                     factor = 584.6;
//                     break; 
//                 case TURB: 
//                     factor = 584.6;
//                     break; 
//                 case COMP: 
//                     factor = 584.6;
//                     break; 
//                 default: 
//                     factor = 584.6;
//                     break; 
//             }
//             break;
//         case 2013: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 567.3;
//                     break; 
//                 case HTEX: 
//                     factor = 567.3;
//                     break; 
//                 case TURB: 
//                     factor = 567.3;
//                     break; 
//                 case COMP: 
//                     factor = 567.3;
//                     break; 
//                 default: 
//                     factor = 567.3;
//                     break; 
//             }
//             break; 
//         case 2014: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 576.1;
//                     break; 
//                 case HTEX: 
//                     factor = 576.1;
//                     break; 
//                 case TURB: 
//                     factor = 576.1;
//                     break; 
//                 case COMP: 
//                     factor = 576.1;
//                     break; 
//                 default: 
//                     factor = 576.1;
//                     break; 
//             }
//             break;
//         case 2015: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 556.8;
//                     break; 
//                 case HTEX: 
//                     factor = 556.8;
//                     break; 
//                 case TURB: 
//                     factor = 556.8;
//                     break; 
//                 case COMP: 
//                     factor = 556.8;
//                     break; 
//                 default: 
//                     factor = 556.8;
//                     break; 
//             }
//             break;
//         case 2016: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 541.7;
//                     break; 
//                 case HTEX: 
//                     factor = 541.7;
//                     break; 
//                 case TURB: 
//                     factor = 541.7;
//                     break; 
//                 case COMP: 
//                     factor = 541.7;
//                     break; 
//                 default: 
//                     factor = 541.7;
//                     break; 
//             }
//             break;
//         case 2017: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 567.5;
//                     break; 
//                 case HTEX: 
//                     factor = 567.5;
//                     break; 
//                 case TURB: 
//                     factor = 567.5;
//                     break; 
//                 case COMP: 
//                     factor = 567.5;
//                     break; 
//                 default: 
//                     factor = 567.5;
//                     break; 
//             }
//             break;
//         case 2018: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 603.1;
//                     break; 
//                 case HTEX: 
//                     factor = 603.1;
//                     break; 
//                 case TURB: 
//                     factor = 603.1;
//                     break; 
//                 case COMP: 
//                     factor = 603.1;
//                     break; 
//                 default: 
//                     factor = 603.1;
//                     break; 
//             }
//             break;
//         case 2019: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 607.5;
//                     break; 
//                 case HTEX: 
//                     factor = 607.5;
//                     break; 
//                 case TURB: 
//                     factor = 607.5;
//                     break; 
//                 case COMP: 
//                     factor = 607.5;
//                     break; 
//                 default: 
//                     factor = 607.5;
//                     break; 
//             }
//             break;
//         case 2020: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 596.2;
//                     break; 
//                 case HTEX: 
//                     factor = 596.2;
//                     break; 
//                 case TURB: 
//                     factor = 596.2;
//                     break; 
//                 case COMP: 
//                     factor = 596.2;
//                     break; 
//                 default: 
//                     factor = 596.2;
//                     break; 
//             }
//             break;
//         case 2021: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 708.8;
//                     break; 
//                 case HTEX: 
//                     factor = 708.8;
//                     break; 
//                 case TURB: 
//                     factor = 708.8;
//                     break; 
//                 case COMP: 
//                     factor = 708.8;
//                     break; 
//                 default: 
//                     factor = 708.8;
//                     break; 
//             }
//             break;
//         case 2022: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 816.0;
//                     break; 
//                 case HTEX: 
//                     factor = 816.0;
//                     break; 
//                 case TURB: 
//                     factor = 816.0;
//                     break; 
//                 case COMP: 
//                     factor = 816.0;
//                     break; 
//                 default: 
//                     factor = 816.0;
//                     break; 
//             }
//             break;
//         case 2023: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 797.9;
//                     break; 
//                 case HTEX: 
//                     factor = 797.9;
//                     break; 
//                 case TURB: 
//                     factor = 797.9;
//                     break; 
//                 case COMP: 
//                     factor = 797.9;
//                     break; 
//                 default: 
//                     factor = 797.9;
//                     break; 
//             }
//             break;
//         case 2024: 
//             switch (type) {
//                 case PIPE: 
//                     factor = 795.4;
//                     break; 
//                 case HTEX: 
//                     factor = 795.4;
//                     break; 
//                 case TURB: 
//                     factor = 795.4;
//                     break; 
//                 case COMP: 
//                     factor = 795.4;
//                     break; 
//                 default: 
//                     factor = 795.4;
//                     break; 
//             }
//             break;
//         default:
//             break;
//     }
// }

double CEPCI(int year, int type) {
    /*
    Taken from the Chemical Engineering Plant Cost Index (CEPCI).
    Each dollar value is corrected from the first month of the given
    year to the first month of 2024.
    
    https://www.bls.gov/data/inflation_calculator.htm
    */

    const int BASE = 0;
    const int PIPE = 1; // Piping, Inventory, etc. 
    const int HTEX = 2; // Heat Exchangers
    const int TURB = 3; // Turbine
    const int COMP = 4; // Compressor
    const int LABR = 5; 
    const int LAND = 6; 
    const int LIFT = 7; 
    const int TANK = 8; 

    static const std::unordered_map<int, std::unordered_map<int, double>> CEPCI_lookup = {
        {2012, {{BASE, 584.6}, {PIPE, 584.6}, {HTEX, 584.6}, {TURB, 584.6}, {COMP, 584.6}, {LABR, 584.6}, {LAND, 584.6}, {LIFT, 584.6}, {TANK, 584.6}}},
        {2013, {{BASE, 567.3}, {PIPE, 567.3}, {HTEX, 567.3}, {TURB, 567.3}, {COMP, 567.3}, {LABR, 567.3}, {LAND, 567.3}, {LIFT, 567.3}, {TANK, 567.3}}},
        {2014, {{BASE, 576.1}, {PIPE, 576.1}, {HTEX, 576.1}, {TURB, 576.1}, {COMP, 576.1}, {LABR, 576.1}, {LAND, 576.1}, {LIFT, 576.1}, {TANK, 576.1}}},
        {2015, {{BASE, 556.8}, {PIPE, 556.8}, {HTEX, 556.8}, {TURB, 556.8}, {COMP, 556.8}, {LABR, 556.8}, {LAND, 556.8}, {LIFT, 556.8}, {TANK, 556.8}}},
        {2016, {{BASE, 541.7}, {PIPE, 541.7}, {HTEX, 541.7}, {TURB, 541.7}, {COMP, 541.7}, {LABR, 541.7}, {LAND, 541.7}, {LIFT, 541.7}, {TANK, 541.7}}},
        {2017, {{BASE, 567.5}, {PIPE, 567.5}, {HTEX, 567.5}, {TURB, 567.5}, {COMP, 567.5}, {LABR, 567.5}, {LAND, 567.5}, {LIFT, 567.5}, {TANK, 567.5}}},
        {2018, {{BASE, 603.1}, {PIPE, 603.1}, {HTEX, 603.1}, {TURB, 603.1}, {COMP, 603.1}, {LABR, 603.1}, {LAND, 603.1}, {LIFT, 603.1}, {TANK, 603.1}}},
        {2019, {{BASE, 607.5}, {PIPE, 607.5}, {HTEX, 607.5}, {TURB, 607.5}, {COMP, 607.5}, {LABR, 607.5}, {LAND, 607.5}, {LIFT, 607.5}, {TANK, 607.5}}},
        {2020, {{BASE, 596.2}, {PIPE, 596.2}, {HTEX, 596.2}, {TURB, 596.2}, {COMP, 596.2}, {LABR, 596.2}, {LAND, 596.2}, {LIFT, 596.2}, {TANK, 596.2}}},
        {2021, {{BASE, 708.8}, {PIPE, 708.8}, {HTEX, 708.8}, {TURB, 708.8}, {COMP, 708.8}, {LABR, 708.8}, {LAND, 708.8}, {LIFT, 708.8}, {TANK, 708.8}}},
        {2022, {{BASE, 816.0}, {PIPE, 816.0}, {HTEX, 816.0}, {TURB, 816.0}, {COMP, 816.0}, {LABR, 816.0}, {LAND, 816.0}, {LIFT, 816.0}, {TANK, 816.0}}},
        {2023, {{BASE, 797.9}, {PIPE, 797.9}, {HTEX, 797.9}, {TURB, 797.9}, {COMP, 797.9}, {LABR, 797.9}, {LAND, 797.9}, {LIFT, 797.9}, {TANK, 797.9}}},
        {2024, {{BASE, 795.4}, {PIPE, 795.4}, {HTEX, 795.4}, {TURB, 795.4}, {COMP, 795.4}, {LABR, 795.4}, {LAND, 795.4}, {LIFT, 795.4}, {TANK, 795.4}}}
    };

    // calculating CEPCI for given year and equipment type
    double factor = CEPCI_lookup.at(year).find(type)->second; 

    // calculating 2024 CEPCI for given equipment type
    double baseln = CEPCI_lookup.at(2024).find(type)->second; 

    return baseln / factor; 
}



